#!/usr/bin/env bash
# Launcher behind the "train" VSCode task.
#
#   run_training.sh <config> <use_tmux: Sim|Não> <resume_choice>
#
# <config> is a path, or a bare file name looked up in EXPERIMENTS_ROOT/configs/
# (bootstrapper.py decides where that is).
#
# All decisions arrive as arguments, answered client-side by VSCode's pick
# lists BEFORE this script runs -- no interactive `read` ever blocks inside a
# shell that may be a child of a flaky SSH session.
#
# Resume detection: the single signal is the file <work_dir>/last_checkpoint,
# written by tools/train.py after every saved checkpoint. `--resume` is only
# passed when the user asked for it AND that file exists.
#
# TensorBoard: train.py always writes event files (--tensorboard on); the
# dashboard itself is the separate "tensorboard" task (run_tensorboard.sh),
# started whenever you want, before or during the training.
set -e

CONFIG_ARG="$1"
USE_TMUX="$2"
RESUME_CHOICE="$3"
TOOLING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$CONFIG_ARG" ]; then
    echo "usage: run_training.sh <config> <Sim|Não> <resume_choice>"
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mae_st_test

# Let the tooling resolve the config (bare name -> configs/) and ask the
# config for its work_dir; both come from bootstrapper/experiment, so this
# script holds no paths of its own.
mapfile -t RESOLVED < <(python - "$TOOLING_DIR" "$CONFIG_ARG" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from custom import experiment
cfg = experiment.load_config(sys.argv[2])
print(cfg["__config_path__"])
print(cfg["work_dir"])
PY
)
CONFIG="${RESOLVED[0]}"
WORK_DIR="${RESOLVED[1]}"
echo "config:   $CONFIG"
echo "work_dir: $WORK_DIR"

RESUME_FLAG=""
if [ "$RESUME_CHOICE" = "Continuar (--resume)" ]; then
    if [ -f "$WORK_DIR/last_checkpoint" ]; then
        RESUME_FLAG="--resume"
        echo "Resuming from: $(cat "$WORK_DIR/last_checkpoint")"
    else
        echo "Note: 'Continuar' was chosen, but no last_checkpoint was found in $WORK_DIR -- starting fresh instead."
    fi
elif [ -f "$WORK_DIR/last_checkpoint" ]; then
    echo "Note: an interrupted run WAS found at $WORK_DIR, but 'Reiniciar do zero' was chosen -- starting fresh (old checkpoints stay until pruned)."
fi

# Inner script in a temp file: avoids quoting the command through tmux.
INNER_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/mae_st_train_inner.XXXXXX.sh")
cat > "$INNER_SCRIPT" <<EOF
#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mae_st_test
python -u "$TOOLING_DIR/tools/train.py" "$CONFIG" $RESUME_FLAG --tensorboard on
echo
echo "[training process ended -- press Enter to close]"
read
EOF

CONFIG_BASENAME=$(basename "$CONFIG" .py)
if [ "$USE_TMUX" = "Sim" ]; then
    SESSION_NAME="${CONFIG_BASENAME}-$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION_NAME" "bash '$INNER_SCRIPT'"
    echo "Training started in tmux session: $SESSION_NAME"
    echo "  attach:  tmux attach -t $SESSION_NAME"
    echo "  detach:  Ctrl+B, D"
else
    bash "$INNER_SCRIPT"
    rm -f "$INNER_SCRIPT"
fi
