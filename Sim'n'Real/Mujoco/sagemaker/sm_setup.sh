#!/usr/bin/env bash
# Re-exec under bash if invoked as `sh script.sh`: `set -o pipefail` is a
# bash builtin and dash/ksh reject it with "invalid option name".
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# STEP 1 -- run ONCE.   bash sm_setup.sh
#
# Inspects the instance, tells you what to expect, and writes sm_config.env
# with the batch size and worker count that suit THIS box. sm_train.sh reads it.
#
# PERSISTENCE: on a classic SageMaker Notebook Instance only
# /home/ec2-user/SageMaker survives a stop/start -- $HOME does NOT. Everything
# that must outlive a stop (venv, HF cache, checkpoints, logs) goes under $ROOT.
set -euo pipefail

if [ -d /home/ec2-user/SageMaker ]; then ROOT=/home/ec2-user/SageMaker; else ROOT="$HOME"; fi
ROOT="${SKYGRIP_ROOT:-$ROOT}"
NEED_GB=25
VENV="$ROOT/skyenv"
export HF_HOME="$ROOT/.cache/huggingface"

echo "============================================================"
echo " instance"
echo "============================================================"
ITYPE=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true)
[ -z "$ITYPE" ] && ITYPE=$(python -c "import json;print(json.load(open('/opt/ml/metadata/resource-metadata.json')).get('InstanceType','?'))" 2>/dev/null || echo "unknown")
NCPU=$(nproc)
RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "?")
echo "type       : $ITYPE"
echo "vCPUs      : $NCPU"
echo "RAM        : ${RAM_GB} GB"
echo "persistent : $ROOT"
[ "$ROOT" = "$HOME" ] && echo "  NOTE: \$HOME is NOT persistent on a classic Notebook Instance."

echo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { echo "STOP: no nvidia-smi -- this is not a GPU instance."; exit 1; }
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo
df -h "$ROOT" | tail -1
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
# If the 14 GB model is already cached the remaining need is CHECKPOINT space,
# not download space -- the old flat 25 GB gate stopped for the wrong reason.
CACHED=0
[ -d "$HF_HOME/hub/models--lerobot--pi0_base" ] && CACHED=1
if [ "$CACHED" = "1" ]; then
  echo "pi0_base already cached -- remaining need is CHECKPOINT space."
  echo "A pi0 checkpoint is ~6-9 GB (6.3 GB of weights + optimizer state)."
  echo "  ${AVAIL_GB} GB free -> room for roughly $((AVAIL_GB/8)) checkpoint(s)."
  if [ "$AVAIL_GB" -lt 20 ]; then
    echo "  RUN THE REAPER next to training or you WILL hit ENOSPC hours in:"
    echo "      bash sm_reap.sh s1 2 &"
    echo "  Better still: raise the EBS volume to >= 100 GB."
  fi
else
  if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    echo
    echo "STOP: ${AVAIL_GB} GB free, need ~${NEED_GB} GB (pi0_base is 14 GB, plus"
    echo "~2 GB data and multi-GB checkpoints). Raise the EBS volume, then re-run."
    exit 1
  fi
fi

# ---------------------------------------------------------------- verdict ----
echo
echo "============================================================"
echo " what to expect on this box"
echo "============================================================"
# batch size from VRAM. Frozen-trunk pi0 needs ~15-20 GB at batch 8.
if   [ "$GPU_MB" -ge 38000 ]; then BS=16; GC=false
elif [ "$GPU_MB" -ge 22000 ]; then BS=8;  GC=false
elif [ "$GPU_MB" -ge 15000 ]; then BS=4;  GC=true
else                               BS=2;  GC=true
fi
WORKERS=$(( NCPU - 1 )); [ "$WORKERS" -lt 1 ] && WORKERS=1

echo "GPU        : $GPU_NAME (${GPU_MB} MB)"
echo "  -> batch_size=$BS, gradient_checkpointing=$GC"
[ "$GPU_MB" -lt 22000 ] && echo "     (tight: frozen-trunk pi0 wants ~15-20 GB at batch 8)"

echo
echo "DATA LOADING -- the thing most likely to waste your time:"
echo "  This dataset is VIDEO-backed (298k frames x 3 camera streams, decoded"
echo "  on the fly). Decoding is CPU work, so too few cores starves the GPU."
if   [ "$NCPU" -le 4 ]; then
  echo "  $NCPU vCPUs: EXPECT TO BE DECODE-BOUND. The GPU will idle waiting for"
  echo "  frames and the run may take several times longer than the GPU implies."
  echo "  Consider a larger instance (ml.g5.2xlarge+) or run on Colab instead."
elif [ "$NCPU" -le 8 ]; then
  echo "  $NCPU vCPUs: borderline. Watch GPU utilisation in the first 10 minutes."
else
  echo "  $NCPU vCPUs: should keep the GPU fed."
fi
echo "  -> num_workers=$WORKERS"

cat > "$ROOT/sm_config.env" <<EOF
# written by sm_setup.sh -- sm_train.sh reads this
SKYGRIP_ROOT=$ROOT
BATCH_SIZE=$BS
NUM_WORKERS=$WORKERS
GRAD_CKPT=$GC
EOF
echo
echo "wrote $ROOT/sm_config.env"

# ------------------------------------------------------------------ deps ----
echo
echo "============================================================"
echo " environment"
echo "============================================================"
mkdir -p "$HF_HOME"
grep -q 'SKYGRIP_ROOT' "$HOME/.bashrc" 2>/dev/null || {
  echo "export SKYGRIP_ROOT=$ROOT"                >> "$HOME/.bashrc"
  echo "export HF_HOME=$ROOT/.cache/huggingface"  >> "$HOME/.bashrc"
}

if python -c "import lerobot,sys; sys.exit(0 if lerobot.__version__.startswith('0.6') else 1)" 2>/dev/null; then
  LOC=$(python -c "import lerobot;print(lerobot.__file__)")
  echo "reusing lerobot $(python -c 'import lerobot;print(lerobot.__version__)')"
  echo "  from $LOC"
  case "$LOC" in
    /home/ec2-user/.local/*)
      echo "  WARNING: that path is NOT persistent -- it disappears on a stop/start."
      echo "  Fine for now; re-run this script after any instance restart." ;;
  esac
  PY=python
else
  [ -d "$VENV" ] || python -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet "lerobot==0.6.0"
  PY="$VENV/bin/python"
  echo "created venv at $VENV (persistent)"
fi
command -v tmux >/dev/null || sudo yum install -y tmux 2>/dev/null || \
  echo "NOTE: no tmux; sm_train.sh falls back to setsid+nohup"

echo
$PY - <<'PY'
import sys, torch, lerobot
print("python ", sys.version.split()[0])
print("torch  ", torch.__version__)
print("lerobot", lerobot.__version__)
ok = torch.cuda.is_available()
print("cuda   ", ok, torch.cuda.get_device_name(0) if ok else "")
if not ok:
    raise SystemExit("STOP: torch cannot see the GPU -- image/driver mismatch.")
PY

echo
echo "============================================================"
echo " next"
echo "============================================================"
echo "  hf auth login          # if not already logged in"
echo "  python sm_pull.py      # pulls pi0_base (14 GB) + dataset"
echo "  bash sm_train.sh s1    # launches detached, using sm_config.env"
