#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

branch="${1:?usage: $0 calibration|a0diag|k0|a0 seed [arap_weight]}"
seed="${2:-0}"
arap_weight="${3:-0}"
scene=c1_sigmoid2_t4_v4
case "$branch" in
    calibration)
        tag="arap_calibration_monitor_seed${seed}"
        regularizer=knn99_arap_monitor
        iterations=10000
        arap_weight=1.0
        full_eval=0
        ;;
    a0diag)
        weight_tag="${arap_weight//./p}"
        tag="arap_a0_stability_w${weight_tag}_seed${seed}"
        regularizer=arap
        iterations=10000
        full_eval=0
        ;;
    k0)
        tag="arap_k0_seed${seed}"
        regularizer=knn99
        iterations=30000
        arap_weight=0
        full_eval=1
        ;;
    a0)
        weight_tag="${arap_weight//./p}"
        tag="arap_a0_w${weight_tag}_seed${seed}"
        regularizer=arap
        iterations=30000
        full_eval=1
        ;;
    *)
        echo "unknown branch: $branch" >&2
        exit 2
        ;;
esac

experiment="${tag}/${scene}"
path="output/${experiment}"
log_dir="logs/${experiment}"
if [[ -e "$path" || -e "$log_dir/status.txt" ]]; then
    echo "run already exists: $experiment" >&2
    exit 1
fi
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/pipeline.log") 2>&1
current=train
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$log_dir/status.txt"; exit "$code"' ERR

printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
/usr/bin/time -v python -u train.py \
    -s "data/coloncrafter/${scene}" --port "$((6100 + seed))" \
    --expname "$experiment" \
    --deformation_mode node --num_nodes 2000 --node_k 4 \
    --node_position_mode direct --node_position_normalization bbox \
    --node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1" \
    --node_mlp_width 128 --node_mlp_depth 3 \
    --color_model node_residual_mlp --node_appearance_dim 8 \
    --node_appearance_graph_mode fixed_initial \
    --color_residual_scale 0.5 --appearance_seed 0 --lambda_res 0 \
    --coarse_iterations 100 --iterations "$iterations" \
    --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 \
    --col_smooth_weight 0.0001 --tv_weight 0.001 \
    --densify_from_iter 500000 --pruning_from_iter 500000 \
    --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000 \
    --gaussian_knn_reference output/adaptive_deformation_matched_knn99.pth \
    --geometry_regularizer "$regularizer" --arap_weight "$arap_weight" \
    --arap_k 6 --arap_eval_k 8 --arap_seed 1907 \
    --arap_gradient_audit_iterations 2000,5000,10000 \
    --experiment_seed "$seed" \
    --adaptive_node_mode none \
    --save_iterations 1 2000 5000 10000 20000 25000

current=audit
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
python tools/evaluate_arap_checkpoint.py "$path" --iteration "$iterations" \
    --evaluation-graph output/arap_backbone_eval_k8.pth --reload-audit

if [[ "$full_eval" == 1 ]]; then
    current=evaluate
    printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
    python render.py -m "$path" -s "data/coloncrafter/${scene}" \
        --iteration 30000 --skip_train --skip_video --eval
    python metrics.py -m "$path"
    python tools/benchmark_adaptive_render.py "$path" --iteration 30000 --repeats 3
fi

printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "$log_dir/status.txt"
