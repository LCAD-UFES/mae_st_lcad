#!/usr/bin/env bash
# TensorBoard over ALL work_dirs, in the FOREGROUND of the calling terminal.
#
#   run_tensorboard.sh [logdir] [port]      defaults: EXPERIMENTS_ROOT/work_dirs, 6006
#
# Deliberately not detached (no tmux): run from the "tensorboard" VSCode task
# it is stoppable from the task panel / terminal trash icon / Ctrl+C, and
# VSCode's Remote-SSH detects the port and offers to open it in the browser
# (auto port forwarding). The server binds to localhost only -- TensorBoard
# has no authentication -- so the safe remote access paths are that
# forwarding or your own tunnel from the local machine, which survives
# editor hiccups:
#
#   autossh -M 0 -N -L 6006:localhost:6006 <user>@<host>
#
# train.py writes the event files regardless of whether this is running, so
# the dashboard can be started before, during or after a training and will
# show every run under work_dirs (in progress or finished) for comparison.
set -e

TOOLING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${2:-6006}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mae_st_test

if [ -n "$1" ]; then
    LOGDIR="$1"
else
    LOGDIR=$(python -c "import sys; sys.path.insert(0, '$TOOLING_DIR'); import bootstrapper; print(bootstrapper.WORK_DIRS)")
fi

echo "TensorBoard over $LOGDIR on http://localhost:$PORT  (stop: Ctrl+C or kill this terminal)"
exec tensorboard --logdir "$LOGDIR" --port "$PORT" --reload_interval 10
