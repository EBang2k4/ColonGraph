#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

log=logs/arap_revised_phase_supervisor.log
status=logs/arap_revised_phase_supervisor.status
mkdir -p logs
exec > >(tee -a "$log") 2>&1
current=calibration10k
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$status"; exit "$code"' ERR

printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
./run_arap_revised_10k.sh calibration

current=plan_lambda
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python tools/plan_arap_revised.py propose
mapfile -t candidate_weights < <(python -c \
    'import json; p=json.load(open("output/arap_revised_phase/calibration_plan.json")); print("\n".join(x["weight_text"] for x in p["candidates"]))')

current=stability10k
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
for weight in "${candidate_weights[@]}"; do
    ./run_arap_revised_10k.sh stability "$weight"
done

current=select_stable_lambda
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python tools/plan_arap_revised.py select

printf '%s REVISED_STABILITY_COMPLETED_STOP_BEFORE_30K\n' \
    "$(date --iso-8601=seconds)" | tee "$status"
