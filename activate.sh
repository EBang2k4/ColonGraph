#!/usr/bin/env bash
# Source this file, or use run.sh.
U7_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
U7_CONDA_PREFIX="${COLONGRAPH_CONDA_PREFIX:-$U7_PACKAGE_ROOT/../.env}"
if [ "${CONDA_PREFIX:-}" != "$(readlink -f "$U7_CONDA_PREFIX")" ]; then
    U7_CONDA_BASE="${COLONGRAPH_CONDA_BASE:-$U7_PACKAGE_ROOT/../.tools/miniconda}"
    if [ -f "$U7_CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$U7_CONDA_BASE/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo 'Conda not found. Set COLONGRAPH_CONDA_BASE and COLONGRAPH_CONDA_PREFIX.' >&2
        return 1
    fi
    conda activate "$U7_CONDA_PREFIX" || return 1
fi
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset U7_CONDA_PREFIX U7_CONDA_BASE
