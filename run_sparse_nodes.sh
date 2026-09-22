#!/usr/bin/env bash
# Arguments: SCENE ITERATIONS RUN_TAG NUM_NODES KNN_WEIGHT NODE_POSITION_MODE NODE_POSITION_NORMALIZATION.
set -Eeuo pipefail
cd /mnt/storage/Manh/ColonSplat
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat
scene="${1:-c1_sigmoid2_t4_v4}"
iterations="${2:-30000}"
num_nodes="${4:-2000}"
knn_weight="${5:-0.01}"
node_position_mode="${6:-direct}"
node_position_normalization="${7:-none}"
tag="${3:-baseline1_nodes${num_nodes}}"
[[ "$scene" =~ ^c[12]_[a-z0-9]+_t[1-4]_v4$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ && "$iterations" =~ ^[0-9]+$ && "$num_nodes" =~ ^[1-9][0-9]*$ && "$knn_weight" =~ ^[0-9]+([.][0-9]+)?$ ]]
[[ "$node_position_mode" == direct || "$node_position_mode" == local_frame ]]
[[ "$node_position_normalization" == none || "$node_position_normalization" == bbox || "$node_position_normalization" == raw_only || "$node_position_normalization" == pe_only ]]
experiment="${tag}/${scene}"
log_dir="logs/${experiment}"
[[ ! -e "output/${experiment}" && ! -e "${log_dir}/status.txt" ]] || { echo "Run already exists: ${experiment}"; exit 1; }
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/pipeline.log") 2>&1
stage=setup
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$stage" "$code" | tee "${log_dir}/status.txt"; exit "$code"' ERR
run_stage() {
    stage="$1"
    shift
    printf '%s RUNNING %s\n' "$(date --iso-8601=seconds)" "$stage" | tee "${log_dir}/status.txt"
    /usr/bin/time -v "$@" > "${log_dir}/${stage}.log" 2>&1
    printf '%s COMPLETED %s\n' "$(date --iso-8601=seconds)" "$stage" | tee "${log_dir}/status.txt"
}
initialization_args=()
if [[ "$node_position_normalization" != none ]]; then
    initialization_args=(--node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1")
fi
run_stage train python -u train.py -s "data/coloncrafter/${scene}" --port 6020 \
    --expname "$experiment" --deformation_mode node --num_nodes "$num_nodes" --node_k 4 --node_position_mode "$node_position_mode" \
    --node_position_normalization "$node_position_normalization" \
    "${initialization_args[@]}" \
    --node_mlp_width 128 --node_mlp_depth 3 --coarse_iterations 100 --iterations "$iterations" \
    --depth_weight 0.25 --knn_weight "$knn_weight" --dcol_weight 0.005 --col_smooth_weight 0.0001 --tv_weight 0.001 \
    --densify_from_iter 500000 --pruning_from_iter 500000 \
    --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000
run_stage render python -u render.py -s "data/coloncrafter/${scene}" \
    --model_path "output/${experiment}" --skip_video --skip_train --eval
run_stage metrics python -u metrics.py --model_path "output/${experiment}"
run_stage render_lookat python -u render_lookat_cameras.py -s "data/coloncrafter/${scene}" \
    --model_path "output/${experiment}" --skip_video --skip_train --eval
run_stage render_time_shift python -u render_time_shift.py -s "data/coloncrafter/${scene}" \
    --model_path "output/${experiment}" --skip_video --skip_train --eval
run_stage profile python -u tools/profile_deformation.py \
    --baseline "output/test/${scene}_knn0.01_depth0.25_dcol0.005/point_cloud/iteration_30000" \
    --node "output/${experiment}/point_cloud/iteration_${iterations}" \
    --output "output/${experiment}/deformation_benchmark.json"
if [[ "$node_position_mode" == local_frame ]]; then
    run_stage profile_b1 python -u tools/profile_deformation.py \
        --baseline "output/baseline1_nodes2000/${scene}/point_cloud/iteration_30000" \
        --node "output/${experiment}/point_cloud/iteration_${iterations}" \
        --output "output/${experiment}/b1_local_benchmark.json"
fi
if [[ "$node_position_normalization" != none ]]; then
    run_stage profile_bbox python -u tools/profile_deformation.py \
        --baseline "output/baseline1_nodes2000/${scene}/point_cloud/iteration_30000" \
        --node "output/${experiment}/point_cloud/iteration_${iterations}" \
        --output "output/${experiment}/b1_bbox_benchmark.json"
fi
run_stage audit python -u tools/audit_sparse_run.py "$experiment"
if [[ "$node_position_normalization" != none ]]; then
    run_stage audit_bbox python -u tools/audit_bbox_run.py "$experiment"
fi
printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "${log_dir}/status.txt"
