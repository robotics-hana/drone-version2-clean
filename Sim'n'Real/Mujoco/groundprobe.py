"""groundprobe.py -- pre-registered language-grounding probe (E3).

For every VALIDATION pair (two banked episodes sharing one layout,
commands swapped): take each member's FIRST-frame observation, query
the policy with BOTH commands (the pair members' own task strings,
read from the dataset -- no simulator import needed), integrate the
predicted 50-step xy displacement in real units, and classify its
heading by cosine against the directions from the drone spawn to the
two objects (positions from the manifest's recorded layout).

Reported per checkpoint (000000 = the unmodified 047500 baseline):
  - correct_rate: fraction of (observation, own-command) queries
    whose integrated displacement heads toward the commanded object;
  - wrong_rate: heads toward the other object;
  - flip_rate: fraction of observations where SWAPPING the command
    flips the heading between the two objects (the direct measure of
    instruction sensitivity -- a salience-driven policy scores ~0).

Flow noise is drawn once per observation (seed 27182) and REUSED for
both commands and every checkpoint, so comparisons are exactly paired.

usage: python groundprobe.py <ckpt_root> <split.json> \
           <manifest_combined.jsonl> <out.json> [repo_id]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT_ROOT, SPLIT, MANIFEST, OUT = sys.argv[1:5]
REPO_ID = sys.argv[5] if len(sys.argv) > 5 else "hanapasta/airvla_v21"
HORIZON = 50
NOISE_SEED = 27182

val = set(json.load(open(SPLIT))["val"])
rows = [json.loads(line) for line in open(MANIFEST)]
by_idx = {r["combined_episode_index"]: r for r in rows
          if "combined_episode_index" in r}
# validation pair members, grouped by pair_id
pairs = {}
for i, r in by_idx.items():
    if r.get("kind") in ("pairA", "pairB") and i in val:
        pairs.setdefault(r["pair_id"], []).append(i)
pair_eps = sorted(i for members in pairs.values() if len(members) == 2
                  for i in members)
print("val pair episodes:", len(pair_eps), flush=True)

ds = LeRobotDataset(REPO_ID, episodes=pair_eps)
# first dataset row of each episode
first = {}
for j in range(ds.num_frames):
    ep = int(ds.hf_dataset[j]["episode_index"])
    if ep not in first:
        first[ep] = j

# per-observation probe spec: (dataset_index, own task, other task,
# unit dirs to commanded/other object from the spawn)
specs = []
for pid, members in sorted(pairs.items()):
    if len(members) != 2:
        continue
    # full __getitem__ (not raw parquet): the parquet rows store only
    # task_index; the task STRING is joined in by the dataset's
    # getitem (probe-crash fix 2026-09-06)
    tasks = {i: ds[first[i]]["task"] for i in members}
    for i in members:
        r = by_idx[i]
        own = r["obj"]
        other = [k for k in r["layout"]["pos"] if k != own][0]
        s = np.array(r["layout"]["start"][0:2])
        d_own = np.array(r["layout"]["pos"][own]) - s
        d_oth = np.array(r["layout"]["pos"][other]) - s
        d_own /= max(1e-9, np.linalg.norm(d_own))
        d_oth /= max(1e-9, np.linalg.norm(d_oth))
        j = members[0] if i == members[1] else members[1]
        specs.append(dict(idx=first[i], own_task=tasks[i],
                          other_task=tasks[j], d_own=d_own,
                          d_oth=d_oth))

g = torch.Generator().manual_seed(NOISE_SEED)
noises = [torch.randn(1, HORIZON, 32, generator=g) for _ in specs]


def heading(disp, d_own, d_oth):
    n = float(np.linalg.norm(disp))
    if n < 0.02:                        # <2 cm net motion: no verdict
        return None
    return "own" if float(disp @ d_own) >= float(disp @ d_oth) \
        else "other"


results = {}
if Path(OUT).exists():
    results = json.load(open(OUT)).get("results", {})
    print("resuming: %d checkpoints already probed" % len(results),
          flush=True)
ckpts = sorted(p for p in Path(CKPT_ROOT).iterdir()
               if p.name.isdigit())
for ck in ckpts:
    if ck.name in results:
        continue
    pm = ck / "pretrained_model"
    policy = PI0Policy.from_pretrained(str(pm)).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config,
                                         pretrained_path=str(pm))
    correct = wrong = null = flips = 0
    with torch.no_grad():
        for spec, nz in zip(specs, noises):
            item = ds[spec["idx"]]
            base = {k: (v.unsqueeze(0).cuda()
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in item.items()
                    if k.startswith("observation")}
            verdicts = {}
            for which in ("own_task", "other_task"):
                batch = dict(base)
                batch["task"] = [spec[which]]
                batch = pre(batch)
                chunk = policy.predict_action_chunk(
                    batch, noise=nz.to("cuda"))
                chunk = post(chunk)[0].float().cpu().numpy()[:HORIZON]
                disp = chunk[:, 0:2].sum(axis=0)
                verdicts[which] = heading(disp, spec["d_own"],
                                          spec["d_oth"])
            v_own = verdicts["own_task"]
            if v_own == "own":
                correct += 1
            elif v_own == "other":
                wrong += 1
            else:
                null += 1
            # a flip = own-command heads to own object AND
            # other-command heads to the other object (from the SAME
            # observation) -- pure instruction sensitivity
            if v_own == "own" and verdicts["other_task"] == "other":
                flips += 1
    n = len(specs)
    results[ck.name] = dict(n=n, correct_rate=correct / n,
                            wrong_rate=wrong / n, null_rate=null / n,
                            flip_rate=flips / n)
    print("ckpt %s: correct %.2f wrong %.2f null %.2f FLIP %.2f"
          % (ck.name, correct / n, wrong / n, null / n, flips / n),
          flush=True)
    # incremental save (2026-09-06: a node requeue truncated the
    # first probe run mid-sweep and end-only saving lost the rows)
    json.dump(dict(partial=True, noise_seed=NOISE_SEED,
                   results=results), open(OUT, "w"), indent=1)
    del policy
    torch.cuda.empty_cache()

json.dump(dict(noise_seed=NOISE_SEED, results=results),
          open(OUT, "w"), indent=1)
print("GROUNDPROBE-DONE", flush=True)
