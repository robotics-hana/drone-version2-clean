#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# ONE COMMAND: setup -> pull -> train -> reaper.
#
#     bash sm_all.sh [tag]          # default tag s1
#
# Idempotent: re-run it after any instance restart. Steps that are already done
# (model cached, deps installed) are skipped, so it costs seconds the second
# time. Everything lands on the persistent volume.
set -euo pipefail

TAG="${1:-s1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /home/ec2-user/SageMaker ]; then ROOT=/home/ec2-user/SageMaker; else ROOT="$HOME"; fi
ROOT="${SKYGRIP_ROOT:-$ROOT}"
export SKYGRIP_ROOT="$ROOT"
export HF_HOME="$ROOT/.cache/huggingface"

banner() { echo; echo "############ $* ############"; echo; }

banner "1/4  environment"
bash "$HERE/sm_setup.sh"

# sm_setup.sh may have built a venv; use it if present.
[ -d "$ROOT/skyenv" ] && source "$ROOT/skyenv/bin/activate"

banner "2/4  auth"
if python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  python -c "from huggingface_hub import whoami; print('logged in as', whoami()['name'])"
else
  echo "NOT LOGGED IN. Run:  hf auth login --token hf_<your real token>"
  echo "(use the fine-grained token: it does not expire, and it needs"
  echo " 'read access to public gated repos' for pi0's PaliGemma)"
  exit 1
fi

banner "3/4  weights + data"
python "$HERE/sm_pull.py"

banner "4/4  launch"
if pgrep -f "lerobot-train" >/dev/null; then
  echo "a lerobot-train process is ALREADY running:"
  pgrep -fa lerobot-train | head -3
  echo "not launching another. Use sm_monitor.py to watch it."
  exit 0
fi

OUT="$ROOT/pickhold_pi0_frozen_$TAG"
if [ -d "$OUT/checkpoints" ]; then
  echo "checkpoints exist at $OUT -- RESUMING"
  bash "$HERE/sm_resume.sh" "$TAG"
else
  bash "$HERE/sm_train.sh" "$TAG"
fi

# Reaper: a pi0 checkpoint is 6-9 GB and lerobot never prunes. On a small
# volume you hit ENOSPC hours in, so start it whenever space is tight.
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 40 ]; then
  echo
  echo "disk ${AVAIL_GB} GB -- starting the checkpoint reaper (keep 2)"
  setsid nohup bash "$HERE/sm_reap.sh" "$TAG" 2 > "$ROOT/reap_$TAG.log" 2>&1 < /dev/null &
  echo "  reaper log: $ROOT/reap_$TAG.log"
fi

cat <<EOF

============================================================
 running. next:
============================================================
  * paste sm_monitor.py into a NOTEBOOK cell (set TAG = "$TAG")
  * after ~10 min, confirm you are GPU-bound not decode-bound:
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
    utilisation well below 100% => raise NUM_WORKERS in $ROOT/sm_config.env
    and relaunch.
  * when it finishes:  python sm_export.py $TAG

  log:    $ROOT/train_$TAG.log
  output: $OUT
EOF
