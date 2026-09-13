"""relabel_v3.py -- build airvla_v3: the v2 dataset with the policy
given ARM CONTROL supervision (pre-registered 2026-09-13, Hana:
"train a v3 vla, same training demonstrations as v2, but have the
policy control the arm as well").

No recollection: the v2 action layout carries two identically-zero
padded dims (3=droll, 4=dpitch) and every frame records the arm
joints in proprioception (state dims 8,9). This script rewrites
dims 3,4 of every action with the per-step arm-joint deltas
reconstructed from consecutive states within each episode
(last frame of an episode gets 0 -- no successor). Everything else
is byte-identical to airvla_v2; videos are copied untouched.

Mechanics: LeRobot v3 stores frames in data/*.parquet; we copy the
dataset directory, patch the action column episode-safely, refresh
meta/stats.json for the action field, and push as
hanapasta/airvla_v3.

usage (cluster, ONLINE for the push):
  python relabel_v3.py [--dry]   # --dry: patch + verify, no push
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path.home() / "Scratch/hf_cache/lerobot/hanapasta/airvla_v2"
DST = Path.home() / "Scratch/hf_cache/lerobot/hanapasta/airvla_v3"
DRY = "--dry" in sys.argv

if not SRC.exists():
    from huggingface_hub import snapshot_download
    print("downloading airvla_v2 ...", flush=True)
    snapshot_download("hanapasta/airvla_v2", repo_type="dataset",
                      local_dir=SRC)

if DST.exists():
    shutil.rmtree(DST)
print("copying dataset dir ...", flush=True)
shutil.copytree(SRC, DST)

n_patched = 0
arm_mag = []
for pq in sorted((DST / "data").rglob("*.parquet")):
    df = pd.read_parquet(pq)
    act = np.stack(df["action"].to_numpy())
    st = np.stack(df["observation.state"].to_numpy())
    epi = df["episode_index"].to_numpy()
    # per-step arm deltas from consecutive states, within-episode
    d_arm = np.zeros((len(df), 2), dtype=act.dtype)
    same_ep = epi[1:] == epi[:-1]
    d_arm[:-1][same_ep] = (st[1:, 8:10] - st[:-1, 8:10])[same_ep]
    assert np.allclose(act[:, 3:5], 0.0), "dims 3,4 not padded zeros?"
    act[:, 3:5] = d_arm
    df["action"] = list(act)
    df.to_parquet(pq)
    n_patched += len(df)
    arm_mag.append(np.abs(d_arm))
arm_mag = np.concatenate(arm_mag)
print("patched %d frames; |d_arm| mean %.4f p99 %.4f max %.4f"
      % (n_patched, arm_mag.mean(), np.percentile(arm_mag, 99),
         arm_mag.max()), flush=True)

# refresh action stats in meta (lerobot reads normalization from meta)
stats_p = DST / "meta" / "stats.json"
stats = json.loads(stats_p.read_text())
acts = []
for pq in sorted((DST / "data").rglob("*.parquet")):
    acts.append(np.stack(pd.read_parquet(pq)["action"].to_numpy()))
A = np.concatenate(acts)
stats["action"]["mean"] = A.mean(0).tolist()
stats["action"]["std"] = A.std(0).tolist()
stats["action"]["min"] = A.min(0).tolist()
stats["action"]["max"] = A.max(0).tolist()
stats_p.write_text(json.dumps(stats, indent=1))
print("stats refreshed: action std =",
      np.round(A.std(0), 5).tolist(), flush=True)

# sanity: reload one file, confirm dims 3,4 nonzero somewhere
df = pd.read_parquet(sorted((DST / "data").rglob("*.parquet"))[0])
a = np.stack(df["action"].to_numpy())
assert np.abs(a[:, 3:5]).max() > 0, "relabel produced all zeros?"
print("verify OK", flush=True)

if not DRY:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo("hanapasta/airvla_v3", repo_type="dataset",
                    private=True, exist_ok=True)
    api.upload_folder(folder_path=str(DST),
                      repo_id="hanapasta/airvla_v3",
                      repo_type="dataset")
    print("PUSHED hanapasta/airvla_v3", flush=True)
print("RELABEL-V3-DONE", flush=True)
