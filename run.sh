#!/usr/bin/env bash
set -e
U7_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$U7_SCRIPT_ROOT/activate.sh"
cd "$U7_SCRIPT_ROOT"
exec python -u u7.py "$@"
