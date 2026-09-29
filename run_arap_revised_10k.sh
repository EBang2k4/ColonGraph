#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

branch="${1:?usage: $0 calibration|stability [arap_weight]}"
arap_weight="${2:-0}"
scene=c1_sigmoid2_t4_v4
case "$branch" in
    calibration)
        tag=arap_revised_calibration_seed0
        regularizer=knn99_arap_monitor
        arap_weight=1.0
        ;;
    stability)
        weight_tag="${arap_weight//./p}"
        tag="arap_revised_stability_w${weight_tag}_seed0"
        regularizer=arap
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
current=train10k
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$log_dir/status.txt"; exit "$code"' ERR

printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
/usr/bin/time -v python -u train.py \
    -s "data/coloncrafter/${scene}" --port 6120 \
    --expname "$experiment" \
    --deformation_mode node --num_nodes 2000 --node_k 4 \
    --node_position_mode direct --node_position_normalization bbox \
    --node_initialization_reference "output/baseline1_nodes2000/${scene}/point_cloud/iteration_1" \
    --node_mlp_width 128 --node_mlp_depth 3 \
    --color_model node_residual_mlp --node_appearance_dim 8 \
    --node_appearance_graph_mode fixed_initial \
    --color_residual_scale 0.5 --appearance_seed 0 --lambda_res 0 \
    --coarse_iterations 100 --iterations 10000 \
    --depth_weight 0.25 --knn_weight 0.01 \
    --canonical_laplacian_weight 0.01 \
    --dcol_weight 0.005 --col_smooth_weight 0.0001 --tv_weight 0.001 \
    --densify_from_iter 500000 --pruning_from_iter 500000 \
    --densify_from_iter_fine 500000 --pruning_from_iter_fine 500000 \
    --gaussian_knn_reference output/adaptive_deformation_matched_knn99.pth \
    --geometry_regularizer "$regularizer" --arap_weight "$arap_weight" \
    --arap_k 6 --arap_eval_k 8 --arap_seed 1907 \
    --arap_gradient_audit_iterations "" \
    --experiment_seed 0 \
    --adaptive_node_mode none \
    --save_iterations 1 2000 5000

current=multipair_gradient_audit
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
python tools/audit_arap_gradients.py "$path" \
    --iterations 2000 5000 10000 --pairs 16

current=stability_audit
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$log_dir/status.txt"
python tools/evaluate_arap_checkpoint.py "$path" --iteration 10000 \
    --evaluation-graph output/arap_backbone_eval_k8.pth --reload-audit

printf '%s PIPELINE_COMPLETED_10K_ONLY\n' "$(date --iso-8601=seconds)" | tee "$log_dir/status.txt"
