#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# Keep only the newest N checkpoints.   bash sm_reap.sh [tag] [keep] &
#
# WHY THIS IS NEEDED: a pi0 checkpoint is ~6-9 GB (6.3 GB of weights plus
# optimizer state). lerobot writes one every --save_freq steps and does not
# prune, so on a 49 GB volume with ~23 GB free you fill the disk after two or
# three saves and training dies -- usually hours in, with no obvious cause in
# the log beyond an ENOSPC.
#
# Run it in the background next to training:
#     bash sm_reap.sh s1 2 &
# 'last' is a symlink to the newest and is never removed.
set -euo pipefail

if [ -d /home/ec2-user/SageMaker ]; then ROOT=/home/ec2-user/SageMaker; else ROOT="$HOME"; fi
ROOT="${SKYGRIP_ROOT:-$ROOT}"
TAG="${1:-s1}"
KEEP="${2:-2}"
CK="$ROOT/pickhold_pi0_frozen_$TAG/checkpoints"

echo "[reap] watching $CK, keeping newest $KEEP"
while true; do
  if [ -d "$CK" ]; then
    # numeric checkpoint dirs only, oldest first; never touch 'last'
    mapfile -t dirs < <(find "$CK" -maxdepth 1 -type d -regextype posix-extended \
                        -regex '.*/[0-9]+$' -printf '%f\n' 2>/dev/null | sort -n)
    n=${#dirs[@]}
    if [ "$n" -gt "$KEEP" ]; then
      for ((i=0; i<n-KEEP; i++)); do
        d="$CK/${dirs[$i]}"
        # never delete the target of 'last'
        if [ "$(readlink -f "$CK/last" 2>/dev/null)" != "$(readlink -f "$d")" ]; then
          echo "[reap] $(date +%H:%M:%S) removing ${dirs[$i]} ($(du -sh "$d" | cut -f1))"
          rm -rf "$d"
        fi
      done
    fi
    df -h "$ROOT" | tail -1 | awk '{print "[reap] disk: "$4" free ("$5" used)"}'
  fi
  sleep 300
done
