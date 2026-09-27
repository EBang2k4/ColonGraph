#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

scene=c1_sigmoid2_t4_v4
pipeline_log="logs/adaptive_deformation_prune_late_${scene}.log"
status_file="logs/adaptive_deformation_prune_late_${scene}.status"
mkdir -p logs
exec > >(tee -a "$pipeline_log") 2>&1
current=setup
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$status_file"; exit "$code"' ERR

evaluate_run() {
    local tag="$1"
    local path="output/${tag}/${scene}"
    python render.py -m "$path" -s "data/coloncrafter/${scene}" \
        --iteration 30000 --skip_train --skip_video --eval
    python metrics.py -m "$path"
    python tools/benchmark_adaptive_render.py "$path" --iteration 30000 --repeats 3
}

current=c_split_mechanism
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status_file"
./run_adaptive_deformation_graph.sh cdiag "$scene"
python tools/audit_full_adaptive.py \
    "output/adaptive_deformation_c_split_mechanism/${scene}" \
    --branch c --iteration 30000 --mechanism-diagnostic
evaluate_run adaptive_deformation_c_split_mechanism
python tools/report_split_mechanism.py \
    "output/adaptive_deformation_c_split_mechanism/${scene}" --iteration 30000

current=d3_prune_late
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status_file"
./run_adaptive_deformation_graph.sh d3 "$scene"
python tools/audit_full_adaptive.py \
    "output/adaptive_deformation_d3_prune_late/${scene}" \
    --branch d3 --iteration 30000
evaluate_run adaptive_deformation_d3_prune_late

current=report
python tools/report_prune_late.py
printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "$status_file"
