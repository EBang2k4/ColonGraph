#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
source /home/ngocminh/miniconda3/etc/profile.d/conda.sh
conda activate colonsplat

root=output/arap_fixed_phase
log=logs/arap_fixed_phase_supervisor.log
status=logs/arap_fixed_phase_supervisor.status
mkdir -p "$root" logs
exec > >(tee -a "$log") 2>&1
current=wait_calibration
trap 'code=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date --iso-8601=seconds)" "$current" "$code" | tee "$status"; exit "$code"' ERR

printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
calibration_status=logs/arap_calibration_monitor_seed0/c1_sigmoid2_t4_v4/status.txt
while true; do
    if [[ -f "$calibration_status" ]] && grep -q PIPELINE_COMPLETED "$calibration_status"; then
        break
    fi
    if [[ -f "$calibration_status" ]] && grep -q FAILED "$calibration_status"; then
        echo "calibration failed" >&2
        exit 1
    fi
    sleep 30
done

current=plan_lambda
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python tools/plan_arap_calibration.py propose
mapfile -t candidate_weights < <(python -c \
    'import json; p=json.load(open("output/arap_fixed_phase/calibration_plan.json")); print("\n".join(x["weight_text"] for x in p["candidates"]))')

current=stability_diagnostics
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
for weight in "${candidate_weights[@]}"; do
    ./run_arap_experiment.sh a0diag 0 "$weight"
done
python tools/plan_arap_calibration.py select
selected_weight=$(python -c \
    'import json; print(json.load(open("output/arap_fixed_phase/stability_selection.json"))["selected_weight_text"])')

current=paired_fixed_3seeds
printf '%s RUNNING stage=%s lambda=%s\n' "$(date --iso-8601=seconds)" "$current" "$selected_weight" | tee "$status"
for seed in 0 1 2; do
    ./run_arap_experiment.sh k0 "$seed"
    ./run_arap_experiment.sh a0 "$seed" "$selected_weight"
done

current=report
printf '%s RUNNING stage=%s\n' "$(date --iso-8601=seconds)" "$current" | tee "$status"
python tools/report_arap_fixed_seeds.py

printf '%s FIXED_PHASE_COMPLETED_AWAITING_A1_REVIEW\n' "$(date --iso-8601=seconds)" | tee "$status"
