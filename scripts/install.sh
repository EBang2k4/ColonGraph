#!/usr/bin/env bash
# Install Python requirements and compile the bundled CUDA dependencies.
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'Usage: bash scripts/install.sh'
    echo 'Run after conda env create -f environment.yml and conda activate colongraph_m2.'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo 'Unexpected arguments. Use --help.' >&2
    exit 2
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo 'Activate the colongraph_m2 Conda environment first.' >&2
    exit 1
fi
source "$PACKAGE_ROOT/activate.sh"
cd "$PACKAGE_ROOT"
if ! command -v nvcc >/dev/null 2>&1; then
    echo 'CUDA compiler nvcc is missing. Install the toolkit from environment.yml.' >&2
    exit 1
fi
python -m pip install 'setuptools<81' wheel
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-radegs
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-texture
python -m pip install --no-build-isolation ./submodules/simple-knn
python scripts/check_environment.py
