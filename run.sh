#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 2 ]]; then
    echo 'Usage: bash run.sh /path/to/dataset output/scene [--geometry]'
    if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then exit 0; fi
    exit 2
fi
DATASET_PATH="$(realpath "$1")"
RUN_PATH="$(realpath -m "$2")"
shift 2
source "$PACKAGE_ROOT/activate.sh"
cd "$PACKAGE_ROOT"
python -u train.py -s "$DATASET_PATH" -m "$RUN_PATH"
python -u render.py -s "$DATASET_PATH" -m "$RUN_PATH" --skip_train --skip_video
python -u metrics.py -s "$DATASET_PATH" -m "$RUN_PATH" "$@"
python -u render_lookat_cameras.py -s "$DATASET_PATH" -m "$RUN_PATH"
