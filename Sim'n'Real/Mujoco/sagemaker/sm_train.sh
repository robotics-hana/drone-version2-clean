#!/usr/bin/env bash
# Re-exec under bash if invoked as `sh script.sh`: `set -o pipefail` is a
# bash builtin and dash/ksh reject it with "invalid option name".
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# STEP 3 -- launch training DETACHED.   bash sm_train.sh [seed_tag] [batch_size]
#
#   bash sm_train.sh s1        # -> ~/pickhold_pi0_frozen_s1, batch 8
#   bash sm_train.sh s2 16     # a second replicate at batch 16
#
# Runs inside tmux so it survives the browser closing, your laptop sleeping and
# your internet dropping. Watch it from the notebook with sm_monitor.py -- that
# cell is only a viewer and interrupting it does nothing to the job.
#
# What it CANNOT survive is the SageMaker space itself being stopped. That is
# why save_freq is 2000: checkpoints land on the persistent volume, so a stop
# costs you at most 2000 steps and sm_resume.sh picks up from there.
set -euo pipefail

if [ -d /home/ec2-user/SageMaker ]; then ROOT=/home/ec2-user/SageMaker; else ROOT="$HOME"; fi
ROOT="${SKYGRIP_ROOT:-$ROOT}"

# sm_setup.sh inspected the instance and wrote these; CLI args still win.
[ -f "$ROOT/sm_config.env" ] && . "$ROOT/sm_config.env"
TAG="${1:-s1}"
BS="${2:-${BATCH_SIZE:-8}}"
GC="${GRAD_CKPT:-false}"
OUT="$ROOT/pickhold_pi0_frozen_$TAG"
LOG="$ROOT/train_$TAG.log"
VENV="$ROOT/skyenv"
# video-backed dataset: decoding is CPU work, and a fast GPU starved by a
# single-threaded loader is the most common way to waste a g5.
WORKERS="${NUM_WORKERS:-$(( $(nproc) - 1 ))}"; [ "$WORKERS" -lt 1 ] && WORKERS=1

RENAME='{"observation.images.camera3": "observation.images.base_0_rgb", "observation.images.camera1": "observation.images.left_wrist_0_rgb", "observation.images.camera2": "observation.images.right_wrist_0_rgb"}'

read -r -d '' CMD <<EOF || true
[ -d $VENV ] && source $VENV/bin/activate
export HF_HOME=$ROOT/.cache/huggingface
lerobot-train \
  --dataset.repo_id=hanapasta/pick_hold_v4s_train \
  --policy.path=lerobot/pi0_base \
  --rename_map='$RENAME' \
  --output_dir=$OUT \
  --batch_size=$BS \
  --steps=30000 \
  --save_freq=2000 \
  --num_workers=$WORKERS \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=$GC \
  --wandb.enable=false \
  --policy.push_to_hub=false 2>&1 | tee -a $LOG
EOF

echo "root   : $ROOT   (must be the PERSISTENT volume)"
echo "output : $OUT"
echo "log    : $LOG"
echo "batch  : $BS   workers: $WORKERS   (vCPUs: $(nproc))"
echo "frozen : train_expert_only=true  -- only the ~300M action expert trains"
echo "gradckpt: $GC"
echo

if [ -e "$OUT" ]; then
  echo "STOP: $OUT already exists. Use a different tag, or resume with:"
  echo "  bash sm_resume.sh $TAG"
  exit 1
fi

if command -v tmux >/dev/null; then
  tmux new-session -d -s "train_$TAG" "$CMD"
  echo "launched in tmux session 'train_$TAG'"
  echo "  attach : tmux attach -t train_$TAG      (detach: Ctrl-b then d)"
else
  setsid nohup bash -c "$CMD" > /dev/null 2>&1 < /dev/null &
  echo $! > "$ROOT/train_$TAG.pid"
  echo "launched detached, pid $(cat "$ROOT/train_$TAG.pid")"
fi

echo
echo "Now open a notebook and run sm_monitor.py to watch it."
echo
echo "AFTER ~10 MINUTES, check you are GPU-bound and not decode-bound:"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5"
echo "  utilisation well under 100% => raise --num_workers and relaunch."
echo "  CUDA OOM => add --policy.gradient_checkpointing=true (costs ~30% speed,"
echo "  saves several GB) BEFORE reducing batch size."
