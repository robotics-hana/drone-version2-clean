"""v2_split.py -- the pre-registered train/validation split for the v2
dataset: BY COMPLETE EPISODES, never frames (Hana, 2026-09-02).

Mechanism guarantees, not just intent:
  - the split is a partition of EPISODE INDICES; LeRobot's
    `--dataset.episodes` consumes episode indices, so the trainer
    physically cannot see part of an episode;
  - stratified by flavour to Hana's quotas -- picks 288/72 (correctives
    72/18 inside that), nav 192/48 -- with a fixed RNG seed (424242)
    declared in finaldroneresults.md before training;
  - the assignment reads the collection MANIFEST (banked rows in bank
    order = dataset episode order), so flavour labels come from the
    collector's own record.

usage: python v2_split.py <v2_manifest.jsonl> <out_split.json>
"""
import json
import sys

import numpy as np

VAL_FRAC = 0.2   # 20% per flavour -- for the full 600 this reproduces
                 # Hana's exact quotas (std 54, corr 18, nav 48 = 120)
                 # and scales to any size (trial runs included)

manifest, out = sys.argv[1], sys.argv[2]
banked = [json.loads(l) for l in open(manifest)
          if json.loads(l).get("banked")]
kinds = [b["kind"] for b in banked]
idx = {k: [i for i, kk in enumerate(kinds) if kk == k]
       for k in ("std", "corr", "nav")}
print("banked: std %d, corr %d, nav %d, total %d"
      % (len(idx["std"]), len(idx["corr"]), len(idx["nav"]), len(banked)))

rng = np.random.default_rng(424242)
val = []
n_std_val = round(len(idx["std"]) * VAL_FRAC)
n_corr_val = round(len(idx["corr"]) * VAL_FRAC)
n_nav_val = round(len(idx["nav"]) * VAL_FRAC)
for k, n in (("std", n_std_val), ("corr", n_corr_val), ("nav", n_nav_val)):
    pick = rng.choice(idx[k], size=n, replace=False)
    val += [int(i) for i in pick]
val = sorted(val)
train = sorted(set(range(len(banked))) - set(val))
assert not set(train) & set(val), "episode appears in both splits"
assert len(train) + len(val) == len(banked)
split = dict(seed=424242, unit="EPISODES (never frames)",
             train=train, val=val,
             val_by_kind=dict(std=n_std_val, corr=n_corr_val,
                              nav=n_nav_val))
json.dump(split, open(out, "w"), indent=1)
print("split -> %s : train %d / val %d episodes, disjoint verified"
      % (out, len(train), len(val)))
