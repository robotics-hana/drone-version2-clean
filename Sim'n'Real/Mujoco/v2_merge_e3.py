"""v2_merge_e3.py -- E3 pipeline stage 2: merge, split, push.

Merges hanapasta/airvla_v2 (600 eps) + hanapasta/airvla_v2_e3 (the E3
collection) into hanapasta/airvla_v21 via lerobot's aggregate tool
(source order preserved: combined indices 0-599 = the original
episodes, 600+ = E3 episodes in banking order). Then:
  - combined manifest: banked rows of both manifests, in order,
    annotated with combined_episode_index (phantom-guard: a banked
    row superseded by a writer-error row with the same attempt
    number is dropped -- the episode never reached the dataset);
  - extended split: the original 480/120 split VERBATIM (old
    episodes never move between sides), terminal episodes 80/20
    episode-level, paired-command episodes assigned AS UNITS (both
    members of a pair on the same side -- a split pair would leak
    its layout into validation), seed 424243;
  - push airvla_v2_e3 and airvla_v21 to the hub, read-back printed.

usage (from the sim dir, ONLINE, HF_TOKEN_PATH set):
  python v2_merge_e3.py e3_manifest_74000.jsonl e3_split.json \
      e3_manifest_combined.jsonl
"""
import json
import sys

import numpy as np
from huggingface_hub import HfApi
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

E3_MAN, OUT_SPLIT, OUT_MAN = sys.argv[1:4]
SPLIT_SEED = 424243                      # new-episode split only; the
                                         # original split is reused
                                         # verbatim, never redrawn


def banked_rows(path):
    rows = [json.loads(line) for line in open(path)]
    werr = {r["attempt"] for r in rows
            if r.get("reason") == "writer-error"}
    return [r for r in rows if r.get("banked")
            and r["attempt"] not in werr]


v2_rows = banked_rows("v2_manifest_71000.jsonl")
e3_rows = banked_rows(E3_MAN)
assert len(v2_rows) == 600, len(v2_rows)
print("e3 banked:", len(e3_rows), flush=True)

aggregate_datasets(["hanapasta/airvla_v2", "hanapasta/airvla_v2_e3"],
                   "hanapasta/airvla_v21")
ds = LeRobotDataset("hanapasta/airvla_v21")
total = ds.meta.total_episodes
assert total == 600 + len(e3_rows), (total, len(e3_rows))
print("aggregated:", total, "episodes,", ds.meta.total_frames,
      "frames", flush=True)

with open(OUT_MAN, "w") as fh:
    for i, r in enumerate(v2_rows + e3_rows):
        fh.write(json.dumps(dict(r, combined_episode_index=i)) + "\n")

old = json.load(open("v2_split.json"))
rng = np.random.default_rng(SPLIT_SEED)
new_idx = list(range(600, total))
term_eps = [i for i, r in zip(new_idx, e3_rows) if r["kind"] == "term"]
units = {}
for i, r in zip(new_idx, e3_rows):
    if r["kind"] in ("pairA", "pairB"):
        units.setdefault(r["pair_id"], []).append(i)
pair_units = [tuple(v) for v in units.values()]


def split20(items):
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, round(0.2 * len(items)))
    return items[n_val:], items[:n_val]


t_tr, t_va = split20(term_eps)
p_tr_u, p_va_u = split20(pair_units)
p_tr = [i for u in p_tr_u for i in u]
p_va = [i for u in p_va_u for i in u]
train = sorted(old["train"] + t_tr + p_tr)
val = sorted(old["val"] + t_va + p_va)
assert not set(train) & set(val)
assert len(train) + len(val) == total
json.dump(dict(train=train, val=val, seed=SPLIT_SEED,
               note="original 480/120 verbatim; term 80/20; pairs "
                    "split as units (members share a side)"),
          open(OUT_SPLIT, "w"), indent=1)
print("split: train %d val %d (new: term %d/%d eps, pair-units %d/%d)"
      % (len(train), len(val), len(t_tr), len(t_va),
         len(p_tr_u), len(p_va_u)), flush=True)

api = HfApi()
for repo in ("hanapasta/airvla_v2_e3", "hanapasta/airvla_v21"):
    LeRobotDataset(repo).push_to_hub()
    info = api.dataset_info(repo)
    print("PUSH-READBACK %s revision %s files-ok" % (repo,
                                                     info.sha[:12]),
          flush=True)
print("MERGE-DONE", flush=True)
