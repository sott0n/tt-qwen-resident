# source scripts/env.sh — python env and paths for running the tests from the repo root
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TT_METAL_HOME=${TT_METAL_HOME:-$ROOT/tt-metal}
export PYTHONPATH=$ROOT:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}
export ARCH_NAME=blackhole
export MESH_DEVICE=${MESH_DEVICE:-P150x4}
source "$TT_METAL_HOME/python_env/bin/activate"
