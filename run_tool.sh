#!/usr/bin/env bash
# Runs tools/<name>.py inside the conda environment (env.sh); behind the
# compute_loss and plot_loss VSCode tasks.
#   run_tool.sh <name> [args...]
set -e
TOOLING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TOOLING_DIR/env.sh"
exec python -u "$TOOLING_DIR/tools/$1.py" "${@:2}"
