"""valcurve_v2.py -- the validation curve and checkpoint selection for
the v2 training (checklist T3/T4/T5/T6).

For EVERY saved checkpoint: teacher-forced action prediction on the 120
held-out validation episodes -- the policy sees the recorded
observations and its predicted 50-step action chunk is compared against
the expert's recorded actions, in REAL units (post-processor applied;
the normalised-units mistake of behaviour_check.py is documented and
avoided). Flow-matching sampling noise is drawn ONCE per evaluation
window from a seeded generator and REUSED for every checkpoint, so the
curve is exactly paired (the unseeded-variance lesson, PROVENANCE.md).

Outputs (json): per checkpoint -- overall MSE, per-dimension MSE (7),
per-flavour MSE (std/corr/nav) -- plus the T4 selection (lowest overall
validation MSE) printed last. Never sees any test data.

usage: python valcurve_v2.py <ckpt_root> <split.json> <manifest.jsonl> \
           <out.json> [repo_id]
(repo_id defaults to hanapasta/airvla_v2 -- the original sweep; the E3
fine-tune sweep passes hanapasta/airvla_v21 with the combined manifest,
whose per_flavour keys then split old kinds (std/corr/nav) from new
ones (term/pairA/pairB) = the forgetting-vs-learning drift detector.)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT_ROOT, SPLIT, MANIFEST, OUT = sys.argv[1:5]
REPO_ID = sys.argv[5] if len(sys.argv) > 5 else "hanapasta/airvla_v2"
HORIZON = 50
WINDOWS_PER_EP = 6                    # evenly spaced eval windows
NOISE_SEED = 31415

split = json.load(open(SPLIT))
val = split["val"]
banked = [json.loads(l) for l in open(MANIFEST)
          if json.loads(l).get("banked")]
kind_of = {i: banked[i]["kind"] for i in range(len(banked))}

ds = LeRobotDataset(REPO_ID, episodes=val)
print("val episodes:", len(val), "frames:", ds.num_frames, flush=True)

# build the evaluation windows once: (dataset_index, episode, kind)
ep_bounds = {}
for i in range(ds.num_frames):
    ep = int(ds.hf_dataset[i]["episode_index"])
    ep_bounds.setdefault(ep, [i, i])
    ep_bounds[ep][1] = i
windows = []
for ep, (lo, hi) in sorted(ep_bounds.items()):
    usable = hi - lo - HORIZON
    if usable <= 0:
        continue
    for w in range(WINDOWS_PER_EP):
        windows.append((lo + int(usable * w / WINDOWS_PER_EP), ep))
print("windows:", len(windows), flush=True)

# pinned noise per window, reused across checkpoints (paired curve)
g = torch.Generator().manual_seed(NOISE_SEED)
noises = [torch.randn(1, HORIZON, 32, generator=g) for _ in windows]

ckpts = sorted(p for p in Path(CKPT_ROOT).iterdir()
               if p.name.isdigit())
# RESUMABLE: reload a partial output and skip finished checkpoints
# (the first full run was wall-clock killed at 5/12); save after EVERY
# checkpoint, not only at the end
results = {}
if Path(OUT).exists():
    results = json.load(open(OUT)).get("results", {})
    print("resuming: %d checkpoints already scored" % len(results),
          flush=True)
for ck in ckpts:
    if ck.name in results:
        continue
    t0 = time.time()
    pm = ck / "pretrained_model"
    policy = PI0Policy.from_pretrained(str(pm)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config,
                                         pretrained_path=str(pm))
    errs, dims, flavs = [], [], {}
    with torch.no_grad():
        for (idx, ep), nz in zip(windows, noises):
            item = ds[idx]
            batch = {k: (v.unsqueeze(0).cuda()
                         if isinstance(v, torch.Tensor) else v)
                     for k, v in item.items()
                     if k.startswith("observation") or k == "task"}
            batch["task"] = [item["task"]]
            batch = pre(batch)
            chunk = policy.predict_action_chunk(
                batch, noise=nz.to("cuda"))
            chunk = post(chunk)[0].float().cpu().numpy()[:HORIZON]
            # actions come from the parquet table directly --
            # ds[i] would decode 3 video frames per index (150 per
            # window), measured at ~80 min/checkpoint
            gt = np.stack([np.asarray(
                ds.hf_dataset[idx + 1 + t]["action"])
                for t in range(HORIZON)])
            e = (chunk - gt) ** 2
            errs.append(e.mean())
            dims.append(e.mean(0))
            flavs.setdefault(kind_of[ep], []).append(e.mean())
    results[ck.name] = dict(
        mse=float(np.mean(errs)),
        per_dim=[float(x) for x in np.mean(dims, 0)],
        per_flavour={k: float(np.mean(v)) for k, v in flavs.items()},
        n_windows=len(errs), sec=round(time.time() - t0, 1))
    print("ckpt %s: val MSE %.6f (%.0fs)" % (
        ck.name, results[ck.name]["mse"],
        results[ck.name]["sec"]), flush=True)
    json.dump(dict(partial=True, results=results), open(OUT, "w"),
              indent=1)
    del policy
    torch.cuda.empty_cache()

best = min(results, key=lambda k: results[k]["mse"])
out = dict(selection_rule="lowest overall validation action MSE "
                          "(pre-declared, finaldroneresults.md T4)",
           selected=best, noise_seed=NOISE_SEED,
           windows_per_episode=WINDOWS_PER_EP, results=results)
json.dump(out, open(OUT, "w"), indent=1)
print("SELECTED CHECKPOINT:", best, "val MSE",
      results[best]["mse"], flush=True)
print("VALCURVE-DONE")
