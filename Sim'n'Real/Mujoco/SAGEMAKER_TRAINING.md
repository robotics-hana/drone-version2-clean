# Frozen-trunk π0 training on SageMaker (ml.g5 / A10G)

Runbook for training on a SageMaker JupyterLab space, in a way that survives
your laptop losing internet. The pattern is: **the job runs detached in the
terminal, a notebook cell tails its log.** The notebook is only a viewer — if
your browser dies, the training does not.

Versions are pinned to what the dataset was produced with. The dataset is
LeRobot `codebase_version v3.0`, video-backed at 30 fps, so a mismatched
lerobot will fail to read it.

    python 3.12.3 · torch 2.11.0+cu130 · lerobot 0.6.0 · huggingface_hub 1.26.0

---

## Know your hardware before you start

**ml.g5 = 1× NVIDIA A10G, 24 GB VRAM** (not the 40 GB A100). That is enough for
frozen-trunk π0 but not roomy:

| item | approx |
| --- | --- |
| bf16 weights, all 3.3 B params | 6.6 GB |
| grads + Adam state for the ~300 M trainable expert | ~5.5 GB |
| activations at `batch_size=8` | a few GB |
| **total** | **~15–20 GB of 24 GB** |

So start at `batch_size=8`. If you see `CUDA out of memory`, add
`--policy.gradient_checkpointing=true` before reducing batch size — it costs
~30% speed and saves several GB.

**The A10G is a mid-range card.** It has ~600 GB/s bandwidth against the DGX
Spark's ~273 GB/s, so expect it to be faster than the cluster — but it is not
an A100, and I have no measurement on either. Benchmark first (step 5).

---

## Step 0 — publish the dataset (run on `sparks`, once)

SageMaker cannot reach `sparks` (it is behind a cloudflared tunnel), so the Hub
is the transfer path. **Note `private=True`** — without it these repos are
created public.

```python
# on sparks: ~/skyenv/bin/python
from pathlib import Path
from huggingface_hub import upload_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for split in ("train", "eval", "recover"):
    repo = f"hanapasta/pick_hold_v4s_{split}"
    ds = LeRobotDataset(repo)
    ds.push_to_hub(private=True, card_kwargs={})
    # push_to_hub does NOT carry the sidecar, and every interpretability
    # script needs it (decision_frame, relation, band, fingerprints).
    upload_file(path_or_fileobj=str(Path(ds.root) / "episodes_meta.json"),
                path_in_repo="episodes_meta.json",
                repo_id=repo, repo_type="dataset")
    print("pushed", repo)
```

Upload is ~2.9 GB total (train 1.9 GB, eval 673 MB, recover 365 MB).

---

## Step 1 — check disk FIRST (this is the usual failure)

A SageMaker Studio JupyterLab space defaults to **5 GB of EBS**. You need
**~25 GB**: 14 GB for `pi0_base`, ~2 GB for the training split, plus
checkpoints (each π0 checkpoint is several GB).

```bash
df -h /home/sagemaker-user   # or: df -h ~
```

If it is small, stop the space and raise its storage in the space settings
before going further. Discovering this at 90% of a model download is a wasted
hour.

---

## Step 2 — environment (terminal)

```bash
# keep the big caches on the persistent volume, not the container root
export HF_HOME=$HOME/.cache/huggingface
mkdir -p "$HF_HOME"
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc

python -m venv ~/skyenv
source ~/skyenv/bin/activate
pip install --upgrade pip
pip install "lerobot==0.6.0"

python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That must print `True` and `NVIDIA A10G`. If `cuda.is_available()` is False the
image and the driver disagree and nothing below will work.

Then authenticate (your token has `repo.write` and gated-repo read, which π0
needs for PaliGemma):

```bash
hf auth login          # paste the token at the hidden prompt
```

---

## Step 3 — pull data and weights up front

Do this deliberately rather than letting training trigger it, so a network
failure happens *now* and not 30 seconds into a run you thought was started.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
snapshot_download("lerobot/pi0_base")                      # ~14 GB
ds = LeRobotDataset("hanapasta/pick_hold_v4s_train")       # ~1.9 GB
print("dataset:", ds.num_episodes, "episodes,", ds.num_frames, "frames")
hf_hub_download("hanapasta/pick_hold_v4s_train", "episodes_meta.json",
                repo_type="dataset", local_dir=".")
PY
```

Expect `480 episodes, 298294 frames`.

---

## Step 4 — launch detached (terminal)

`tmux` is the right tool: the job keeps running when the browser closes, and
you can reattach.

```bash
tmux new -s train        # if tmux is missing: sudo yum install -y tmux
source ~/skyenv/bin/activate
export HF_HOME=$HOME/.cache/huggingface

lerobot-train \
  --dataset.repo_id=hanapasta/pick_hold_v4s_train \
  --policy.path=lerobot/pi0_base \
  --rename_map='{"observation.images.camera3": "observation.images.base_0_rgb",
                 "observation.images.camera1": "observation.images.left_wrist_0_rgb",
                 "observation.images.camera2": "observation.images.right_wrist_0_rgb"}' \
  --output_dir=$HOME/pickhold_pi0_frozen_s1 \
  --batch_size=8 --steps=30000 --save_freq=2000 \
  --num_workers=4 \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --wandb.enable=false --policy.push_to_hub=false \
  2>&1 | tee -a $HOME/train_s1.log
```

Detach with **Ctrl-b then d**. Reattach later with `tmux attach -t train`.

If `tmux` is unavailable, the equivalent is:

```bash
setsid nohup lerobot-train ... > $HOME/train_s1.log 2>&1 < /dev/null &
echo $! > $HOME/train_s1.pid
```

Two deliberate choices:

- **`--save_freq=2000`, not 5000.** A SageMaker space can be stopped by an idle
  policy or an admin. Checkpoints land on the persistent volume and are what
  makes that survivable, so save often; the cost is disk, not much time.
- **`--num_workers=4`.** Your cluster uses 0, but this dataset is video-backed
  and decoding is CPU work. On a fast GPU with few vCPUs, decode becomes the
  bottleneck. Set this to roughly `vCPUs − 1` (`nproc` to check).

---

## Step 5 — benchmark before committing

Let it run ~10 minutes, then check you are GPU-bound rather than decode-bound:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5
```

**If utilisation is not near 100%, you are starved on data loading** — raise
`--num_workers`. A 30k-step run at the wrong setting can take several times
longer than it should. Get steps/sec from the log and multiply by 30,000 before
you walk away.

---

## Step 6 — watch from the notebook (this is what keeps the space alive)

Put this in a notebook cell. It tails the log the terminal job is writing, and
the running kernel counts as activity against idle-shutdown policies.

```python
import subprocess, time
from pathlib import Path
from IPython.display import clear_output

log = Path.home() / "train_s1.log"
while True:
    tail = subprocess.run(["tail", "-n", "40", str(log)],
                          capture_output=True, text=True).stdout
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    clear_output(wait=True)
    print(f"GPU: {gpu}\n{'-'*70}\n{tail}")
    time.sleep(20)
```

Interrupt the cell whenever you like — **it does not touch the training job**.
Closing the browser is fine too. What you cannot survive is the *space itself*
being stopped, which is what step 6's activity and step 4's `save_freq` are for.

---

## Step 7 — resume after an interruption

```bash
ls $HOME/pickhold_pi0_frozen_s1/checkpoints/     # find the last step
lerobot-train --config_path=$HOME/pickhold_pi0_frozen_s1/checkpoints/last/pretrained_model/train_config.json \
              --resume=true
```

Resume reads the saved config, so you do not retype the flags — and cannot
accidentally resume with different ones.

---

## Step 8 — get the checkpoint somewhere durable

The space's EBS is not a backup. When training finishes:

```bash
python - <<'PY'
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
p = PI0Policy.from_pretrained("<path>/checkpoints/last/pretrained_model")
p.push_to_hub("hanapasta/pickhold-pi0-frozen-s1", private=True)
PY
```

Then the analysis chain in `INTERPRETABILITY_README.md` runs against it —
`probe_extract_pi0.py`, `direction_controls.py`, `ablate_sweeps.py`. Those need
`episodes_meta.json`, which is why step 0 uploads it alongside the data.

---

## Things that will bite, in the order they usually do

1. **5 GB default EBS.** Check before downloading 14 GB of weights.
2. **Decode-bound training.** GPU at 30% means raise `--num_workers`.
3. **Space stopped by idle policy.** Frequent checkpoints + the notebook tail.
4. **OOM at 24 GB.** Add `--policy.gradient_checkpointing=true` before dropping
   batch size — you lose speed rather than gradient quality.
5. **Wrong lerobot version.** The dataset is `v3.0`; pin `lerobot==0.6.0`.
