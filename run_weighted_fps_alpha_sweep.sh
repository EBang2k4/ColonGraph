#!/usr/bin/env bash
# Controlled alpha sweep. Both runs reuse the audited alpha=1 coarse importance map.
set -Eeuo pipefail
cd /mnt/storage/Manh/ColonSplat
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

scene=c1_sigmoid2_t4_v4
importance_reference="output/joint_b1_bbox_r6_weighted_fps/${scene}/weighted_fps_sampling.pth"
log_dir="logs/weighted_fps_alpha_sweep/${scene}"
mkdir -p "$log_dir"
exec 9>"${log_dir}/pipeline.lock"
flock -n 9 || { echo 'Weighted FPS alpha sweep already running'; exit 1; }
[[ -f "$importance_reference" ]] || { echo "Missing audited importance reference: $importance_reference"; exit 1; }

stage=setup
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$stage" "$code" | tee "${log_dir}/status.txt"; exit "$code"' ERR
printf '%s\n' "$$" > "${log_dir}/pipeline.pid"

wait_resources() {
    stage=wait_resources
    while true; do
        local available_kb gpu_free gpu_util
        available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        if (( available_kb >= 7*1024*1024 && gpu_free >= 8000 && gpu_util < 10 )); then
            break
        fi
        printf '%s WAITING_RESOURCES available_kb=%s gpu_free_mb=%s gpu_util=%s required_ram_gib=7\n' \
            "$(date --iso-8601=seconds)" "$available_kb" "$gpu_free" "$gpu_util" | tee "${log_dir}/status.txt"
        sleep 30
    done
}

run_stage() {
    stage="$1"; shift
    printf '%s RUNNING %s\n' "$(date --iso-8601=seconds)" "$stage" | tee "${log_dir}/status.txt"
    /usr/bin/time -v "$@" > "${log_dir}/${stage}.log" 2>&1
    printf '%s COMPLETED %s\n' "$(date --iso-8601=seconds)" "$stage" | tee "${log_dir}/status.txt"
}

train_run() {
    local label="$1" steps="$2" tag="$3" alpha="$4"
    run_stage "$label" python -u train.py -s "data/coloncrafter/${scene}" --port 6020 \
        --expname "$tag" --deformation_mode node --num_nodes 2000 --node_k 4 \
        --node_position_mode direct --node_position_normalization bbox \
        --node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1" \
        --node_mlp_width 128 --node_mlp_depth 3 --node_sampling weighted_fps \
        --node_sampling_alpha "$alpha" --node_sampling_importance_reference "$importance_reference" \
        --color_model node_residual_mlp --node_appearance_dim 8 \
        --color_residual_scale 0.3 --appearance_seed 0 --lambda_res 0 \
        --coarse_iterations 100 --iterations "$steps" \
        --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 \
        --col_smooth_weight 0.0001 --tv_weight 0.001 \
        --densify_from_iter 500000 --pruning_from_iter 500000 \
        --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000
}

run_one() {
    local alpha="$1" suffix="$2"
    local experiment="joint_b1_bbox_r6_weighted_fps_${suffix}/${scene}"
    local smoke="joint_b1_bbox_r6_weighted_fps_${suffix}_smoke20/${scene}"
    [[ ! -e "output/${experiment}" && ! -e "output/${smoke}" ]] || {
        echo "Output already exists: ${experiment} or ${smoke}"; return 1;
    }
    train_run "${suffix}_smoke_train" 20 "$smoke" "$alpha"
    run_stage "${suffix}_smoke_sampling_audit" python -u tools/audit_weighted_fps.py "output/${smoke}"
    run_stage "${suffix}_smoke_joint_audit" python -u tools/audit_joint_r6.py "$smoke"
    wait_resources
    train_run "${suffix}_train" 30000 "$experiment" "$alpha"
    run_stage "${suffix}_sampling_audit" python -u tools/audit_weighted_fps.py "output/${experiment}"
    run_stage "${suffix}_joint_audit" python -u tools/audit_joint_r6.py "$experiment"
    run_stage "${suffix}_render" python -u render.py -s "data/coloncrafter/${scene}" \
        --model_path "output/${experiment}" --skip_video --skip_train --eval
    run_stage "${suffix}_metrics" python -u metrics.py --model_path "output/${experiment}"
    run_stage "${suffix}_profile" python -u tools/profile_deformation.py \
        --baseline "output/joint_b1_bbox_r6/${scene}/point_cloud/iteration_30000" \
        --node "output/${experiment}/point_cloud/iteration_30000" \
        --output "output/${experiment}/standard_vs_weighted_profile.json"
    run_stage "${suffix}_report" python -u tools/report_weighted_fps.py "output/${experiment}"
}

wait_resources
run_stage tests python -m unittest tests.test_weighted_fps tests.test_sparse_nodes tests.test_residual_appearance -v
run_one 0.25 a025
run_one 0.5 a050
run_stage sweep_report python -u tools/report_weighted_fps_sweep.py --scene "$scene"
printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "${log_dir}/status.txt"
