"""STEP 4 -- paste this into a NOTEBOOK CELL and run it.

It tails the log the detached job is writing and shows GPU state. Two jobs:
  * you can watch training without keeping a terminal open;
  * the running kernel counts as activity, which is what stops an idle-shutdown
    policy from stopping the space out from under the job.

It is ONLY a viewer. Interrupting the cell, closing the browser or losing your
internet does nothing to the training process -- that lives in tmux.

    TAG = "s1"   # match the tag you passed to sm_train.sh
"""
import re
import subprocess
import time
from pathlib import Path

import os, pathlib
ROOT = pathlib.Path(os.environ.get("SKYGRIP_ROOT")
                    or ("/home/ec2-user/SageMaker"
                        if os.path.isdir("/home/ec2-user/SageMaker")
                        else pathlib.Path.home()))

from IPython.display import clear_output

TAG = "s1"
REFRESH_S = 20
LOG = ROOT / f"train_{TAG}.log"
OUT = ROOT / f"pickhold_pi0_frozen_{TAG}"
TOTAL_STEPS = 30000

_step = re.compile(r"step[:=\s]+(\d[\d,]*)", re.I)


def _gpu():
    try:
        q = ("utilization.gpu,memory.used,memory.total,temperature.gpu")
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        u, used, tot, temp = [x.strip() for x in r.stdout.strip().split(",")]
        warn = "  <-- DECODE-BOUND? raise --num_workers" if int(u) < 70 else ""
        return f"GPU {u}% | {int(used)/1024:.1f}/{int(tot)/1024:.0f} GB | {temp}C{warn}"
    except Exception as e:  # noqa: BLE001
        return f"GPU query failed: {e}"


def _progress(text):
    hits = _step.findall(text)
    if not hits:
        return ""
    s = int(hits[-1].replace(",", ""))
    pct = 100.0 * s / TOTAL_STEPS
    return f"step {s:,}/{TOTAL_STEPS:,}  ({pct:.1f}%)"


def _alive():
    r = subprocess.run(["pgrep", "-fa", "lerobot-train"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


while True:
    tail = ""
    if LOG.exists():
        tail = subprocess.run(["tail", "-n", "35", str(LOG)],
                              capture_output=True, text=True).stdout
    ckpts = sorted(p.name for p in (OUT / "checkpoints").glob("*")) \
        if (OUT / "checkpoints").exists() else []
    clear_output(wait=True)
    print(f"{_gpu()}")
    print(f"job running: {_alive()}   {_progress(tail)}")
    print(f"checkpoints: {', '.join(ckpts[-6:]) or 'none yet'}")
    print("-" * 78)
    print(tail or f"(no log yet at {LOG})")
    if not _alive() and ckpts:
        print("\n*** lerobot-train is no longer running. If this was not "
              "expected, resume with:  bash sm_resume.sh " + TAG)
        break
    time.sleep(REFRESH_S)
