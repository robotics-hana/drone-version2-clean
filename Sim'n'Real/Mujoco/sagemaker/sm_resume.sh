#!/usr/bin/env bash
# Re-exec under bash if invoked as `sh script.sh`: `set -o pipefail` is a
# bash builtin and dash/ksh reject it with "invalid option name".
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# STEP 5 -- resume after the space was stopped / the job died.
#   bash sm_resume.sh [seed_tag]
#
# Resumes from the saved train_config.json rather than retyping flags, so you
# cannot accidentally continue a run with different hyperparameters -- which
# would quietly invalidate the run without erroring.
set -euo pipefail

if [ -d /home/ec2-user/SageMaker ]; then ROOT=/home/ec2-user/SageMaker; else ROOT="$HOME"; fi
ROOT="${SKYGRIP_ROOT:-$ROOT}"

TAG="${1:-s1}"
OUT="$ROOT/pickhold_pi0_frozen_$TAG"
LOG="$ROOT/train_$TAG.log"
VENV="$ROOT/skyenv"

[ -d "$OUT/checkpoints" ] || { echo "no checkpoints at $OUT"; exit 1; }
echo "checkpoints present:"
ls -1 "$OUT/checkpoints" | tail -6

CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }

if pgrep -f "lerobot-train" >/dev/null; then
  echo "STOP: a lerobot-train process is already running. Check it first:"
  pgrep -fa lerobot-train | head -3
  exit 1
fi

read -r -d '' CMD <<EOF || true
[ -d $VENV ] && source $VENV/bin/activate
export HF_HOME=$ROOT/.cache/huggingface
lerobot-train --config_path=$CFG --resume=true 2>&1 | tee -a $LOG
EOF

if command -v tmux >/dev/null; then
  tmux new-session -d -s "train_$TAG" "$CMD"
  echo "resumed in tmux session 'train_$TAG'"
else
  setsid nohup bash -c "$CMD" > /dev/null 2>&1 < /dev/null &
  echo $! > "$ROOT/train_$TAG.pid"
  echo "resumed detached, pid $(cat "$ROOT/train_$TAG.pid")"
fi
