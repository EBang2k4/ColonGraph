#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

scene=c1_sigmoid2_t4_v4
tag=adaptive_deformation_c_geometry_only_eta05
path="output/${tag}/${scene}"
log="logs/${tag}_${scene}.log"
status="logs/${tag}_${scene}.status"
mkdir -p logs
exec > >(tee -a "$log") 2>&1
current=train
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$status"; exit "$code"' ERR

printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
./run_adaptive_deformation_graph.sh c_geom_eta05 "$scene"

current=audit
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python tools/audit_full_adaptive.py "$path" --branch c --iteration 30000 \
    --fixed-appearance-graph

current=evaluate
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python render.py -m "$path" -s "data/coloncrafter/${scene}" \
    --iteration 30000 --skip_train --skip_video --eval
python metrics.py -m "$path"
python tools/benchmark_adaptive_render.py "$path" --iteration 30000 --repeats 3
python tools/report_geometry_only_split.py

printf '%s PIPELINE_COMPLETED\n' "$(date --iso-8601=seconds)" | tee "$status"
