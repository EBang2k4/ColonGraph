#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

stage="${1:?usage: $0 a|b|s0|s1|c|cdiag|d|d2|d3 [scene]}"
scene="${2:-c1_sigmoid2_t4_v4}"
sampling_args=()
case "$stage" in
    a)
        tag=adaptive_deformation_a_fps2000_fixed
        iterations=30000
        adaptive_mode=none
        adaptive_end=20000
        ;;
    b)
        tag=adaptive_deformation_b_fps2000_importance400_fixed
        iterations=30000
        adaptive_mode=none
        adaptive_end=20000
        sampling_args=(--node_sampling fps_plus_importance
            --node_importance_extra_nodes 400 --node_sampling_gradient_source total)
        ;;
    s0)
        tag=adaptive_deformation_s0_stats
        iterations=5200
        adaptive_mode=stats
        adaptive_end=5000
        ;;
    s1)
        tag=adaptive_deformation_s1_one_split
        iterations=5200
        adaptive_mode=split_once
        adaptive_end=5000
        ;;
    c)
        tag=adaptive_deformation_c_split_only
        iterations=30000
        adaptive_mode=split
        adaptive_end=20000
        ;;
    cdiag)
        tag=adaptive_deformation_c_split_mechanism
        iterations=30000
        adaptive_mode=split
        adaptive_end=20000
        ;;
    d)
        tag=adaptive_deformation_d_split_prune
        iterations=30000
        adaptive_mode=split_prune
        adaptive_end=20000
        ;;
    d2)
        tag=adaptive_deformation_d2_conservative_prune
        iterations=30000
        adaptive_mode=split_prune_conservative
        adaptive_end=20000
        ;;
    d3)
        tag=adaptive_deformation_d3_prune_late
        iterations=30000
        adaptive_mode=split_prune_late
        adaptive_end=25000
        ;;
    *)
        echo "unknown stage: $stage" >&2
        exit 2
        ;;
esac

experiment="${tag}/${scene}"
log_dir="logs/${experiment}"
if [[ -e "output/${experiment}" || -e "${log_dir}/status.txt" ]]; then
    echo "run already exists: ${experiment}" >&2
    exit 1
fi
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/pipeline.log") 2>&1
current=train
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "${log_dir}/status.txt"; exit "$code"' ERR

printf '%s RUNNING %s\n' "$(date --iso-8601=seconds)" "$current" | tee "${log_dir}/status.txt"
/usr/bin/time -v python -u train.py \
    -s "data/coloncrafter/${scene}" --port 6073 \
    --expname "$experiment" \
    --deformation_mode node --num_nodes 2000 --node_k 4 \
    "${sampling_args[@]}" \
    --node_position_mode direct --node_position_normalization bbox \
    --node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1" \
    --node_mlp_width 128 --node_mlp_depth 3 \
    --color_model node_residual_mlp --node_appearance_dim 8 \
    --color_residual_scale 0.3 --appearance_seed 0 --lambda_res 0 \
    --coarse_iterations 100 --iterations "$iterations" \
    --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 \
    --col_smooth_weight 0.0001 --tv_weight 0.001 \
    --densify_from_iter 500000 --pruning_from_iter 500000 \
    --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000 \
    --gaussian_knn_reference output/adaptive_deformation_matched_knn99.pth \
    --adaptive_node_mode "$adaptive_mode" \
    --adaptive_node_start 5000 --adaptive_node_end "$adaptive_end" \
    --adaptive_node_interval 2500 --adaptive_node_max 2400 --adaptive_node_min 1600 \
    --adaptive_node_split_end 20000 --adaptive_node_prune_start 22500 \
    --adaptive_node_max_splits 60 --adaptive_node_max_prunes 40 \
    --adaptive_node_prune_windows 2 --adaptive_node_prune_quantile 0.20 \
    --adaptive_node_prune_redundancy_distance_ratio 1.0 \
    --adaptive_node_prune_redundancy_overlap 0.50 \
    --adaptive_node_prune_motion_error_ratio 0.01 \
    --adaptive_node_prune_safe_error_ratio 0.002 \
    --adaptive_node_score_decay 0.99 \
    --adaptive_node_running_decay 0.9 --adaptive_node_split_z 1.0 \
    --adaptive_node_min_coverage 32 --adaptive_node_min_age 5000 \
    --adaptive_node_weight_threshold 0.05 --adaptive_node_max_split_shock_ratio 0.01 \
    --save_iterations 1 5000 7500 10000 12500 15000 17500 20000 22500 25000
printf '%s COMPLETED %s\n' "$(date --iso-8601=seconds)" "$current" | tee "${log_dir}/status.txt"

printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "${log_dir}/status.txt"
