#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

scene=c1_sigmoid2_t4_v4
pipeline_log="logs/adaptive_deformation_followup_${scene}.log"
mkdir -p logs
exec > >(tee -a "$pipeline_log") 2>&1

evaluate_run() {
    local tag="$1"
    local path="output/${tag}/${scene}"
    python render.py -m "$path" -s "data/coloncrafter/${scene}" \
        --iteration 30000 --skip_train --skip_video --eval
    python metrics.py -m "$path"
    python tools/benchmark_adaptive_render.py "$path" --iteration 30000 --repeats 3
}

./run_adaptive_deformation_graph.sh a "$scene"
python tools/audit_matched_fixed.py \
    "output/adaptive_deformation_a_fps2000_fixed/${scene}" --branch a
evaluate_run adaptive_deformation_a_fps2000_fixed

./run_adaptive_deformation_graph.sh b "$scene"
python tools/audit_matched_fixed.py \
    "output/adaptive_deformation_b_fps2000_importance400_fixed/${scene}" --branch b
evaluate_run adaptive_deformation_b_fps2000_importance400_fixed

./run_adaptive_deformation_graph.sh d2 "$scene"
python tools/audit_full_adaptive.py \
    "output/adaptive_deformation_d2_conservative_prune/${scene}" \
    --branch d2 --iteration 30000
evaluate_run adaptive_deformation_d2_conservative_prune

python tools/report_adaptive_deformation.py
printf '%s FOLLOWUP_PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)"
