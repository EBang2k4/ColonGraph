#!/usr/bin/env bash
# Usage: ./run_joint_b1_bbox_r6.sh [smoke|full] [scene]
set -Eeuo pipefail
cd /mnt/storage/Manh/ColonSplat
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

mode="${1:-smoke}"
scene="${2:-c1_sigmoid2_t4_v4}"
[[ "$mode" == smoke || "$mode" == full ]]
[[ "$scene" =~ ^c[12]_[a-z0-9]+_t[1-4]_v4$ ]]
if [[ "$mode" == smoke ]]; then
    iterations=1000
    tag=joint_b1_bbox_r6_smoke1000
else
    iterations=30000
    tag=joint_b1_bbox_r6
fi
experiment="${tag}/${scene}"
log_dir="logs/${experiment}"
[[ ! -e "output/${experiment}" && ! -e "${log_dir}/status.txt" ]] || {
    echo "Run already exists: ${experiment}"
    exit 1
}
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

run_stage train python -u train.py -s "data/coloncrafter/${scene}" --port 6020 \
    --expname "$experiment" --deformation_mode node --num_nodes 2000 --node_k 4 \
    --node_position_mode direct --node_position_normalization bbox \
    --node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1" \
    --node_mlp_width 128 --node_mlp_depth 3 \
    --color_model node_residual_mlp --node_appearance_dim 8 \
    --color_residual_scale 0.3 --appearance_seed 0 --lambda_res 0 \
    --coarse_iterations 100 --iterations "$iterations" \
    --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 \
    --col_smooth_weight 0.0001 --tv_weight 0.001 \
    --densify_from_iter 500000 --pruning_from_iter 500000 \
    --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000

run_stage audit python -u tools/audit_joint_r6.py "$experiment" --iteration "$iterations"

if [[ "$mode" == full ]]; then
    run_stage render python -u render.py -s "data/coloncrafter/${scene}" \
        --model_path "output/${experiment}" --skip_video --skip_train --eval
    run_stage metrics python -u metrics.py --model_path "output/${experiment}"
    run_stage profile python -u tools/profile_deformation.py \
        --baseline "output/baseline1_bbox_nodes2000/${scene}/point_cloud/iteration_30000" \
        --node "output/${experiment}/point_cloud/iteration_30000" \
        --output "output/${experiment}/b1_bbox_hexplane_vs_joint_r6_profile.json"
fi

printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "${log_dir}/status.txt"
