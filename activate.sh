#!/usr/bin/env bash
# Source this file, or use run.sh. An active Conda environment is preferred.
M2_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${COLONGRAPH_CONDA_PREFIX:-}" ]]; then
    U8_CONDA_TARGET="$COLONGRAPH_CONDA_PREFIX"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    U8_CONDA_TARGET="$CONDA_PREFIX"
elif [[ -d "$M2_PACKAGE_ROOT/../.env/conda-meta" ]]; then
    U8_CONDA_TARGET="$M2_PACKAGE_ROOT/../.env"
else
    U8_CONDA_TARGET="colongraph_m2"
fi
if [[ "${CONDA_PREFIX:-}" != "$U8_CONDA_TARGET" ]]; then
    if [[ -n "${COLONGRAPH_CONDA_BASE:-}" ]]; then
        U8_CONDA_BASE="$COLONGRAPH_CONDA_BASE"
    elif command -v conda >/dev/null 2>&1; then
        U8_CONDA_BASE="$(conda info --base)" || return 1
    else
        U8_CONDA_BASE="$M2_PACKAGE_ROOT/../.tools/miniconda"
    fi
    if [[ ! -f "$U8_CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        echo 'Activate the colongraph_m2 Conda environment first; see README.md.' >&2
        return 1
    fi
    # Conda compiler hooks can reference unset variables under `set -u`.
    U8_HAD_NOUNSET=0
    if [[ $- == *u* ]]; then U8_HAD_NOUNSET=1; fi
    set +u
    U8_ACTIVATION_STATUS=0
    source "$U8_CONDA_BASE/etc/profile.d/conda.sh" && conda activate "$U8_CONDA_TARGET" || U8_ACTIVATION_STATUS=$?
    if [[ "$U8_HAD_NOUNSET" == 1 ]]; then set -u; fi
    if [[ "$U8_ACTIVATION_STATUS" != 0 ]]; then return "$U8_ACTIVATION_STATUS"; fi
    unset U8_HAD_NOUNSET U8_ACTIVATION_STATUS
fi
# Respect a configured toolkit. Conda's CUDA toolkit is the fallback.
if [[ -z "${CUDA_HOME:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
fi
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Torch detects the GPU architecture unless the caller overrides it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset U8_CONDA_TARGET U8_CONDA_BASE
