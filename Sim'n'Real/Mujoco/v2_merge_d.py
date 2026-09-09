"""v2_merge_d.py -- Plan D pipeline stage 2: merge, split, push.

Same machinery as v2_merge_e3.py, one level up the chain: merges
hanapasta/airvla_v21 (910 eps: original 600 + E3) with
hanapasta/airvla_v2_d (the Plan D pairs-only collection, seed 75000)
into hanapasta/airvla_v22 (source order preserved: combined indices
0-909 = airvla_v21 verbatim, 910+ = D episodes in banking order).
Then:
  - combined manifest: the E3-combined manifest rows verbatim + D's
    banked rows (writer-error phantom-guard as before), annotated
    with combined_episode_index;
  - extended split: the airvla_v21 split VERBATIM (episodes never
    move between sides), D pairs assigned AS UNITS (both members of
    a pair on the same side), seed 424244;
  - push airvla_v2_d and airvla_v22 to the hub, read-back printed.

usage (from the sim dir, ONLINE, HF_TOKEN_PATH set):
  python v2_merge_d.py d_manifest_75000.jsonl d_split.json \
      d_manifest_combined.jsonl
"""
import json
import sys

import numpy as np
from huggingface_hub import HfApi
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Amendment 2026-09-08: collection ran 2x slower than the E3-era
# estimate (5.4 min/ep, ETA 52 h > the 30 h wall), so the 300 pairs
# were split across two parallel jobs/repos: airvla_v2_d (seed 75000,
# truncated at its wall) + airvla_v2_d2 (seed 76000, 135 pairs).
# Amendment 2026-09-09: the wall-kill of airvla_v2_d proved
# UNRECOVERABLE -- the LeRobot v3 writer concatenates the whole
# dataset into single parquet files whose footer is written only at
# close, so a hard kill leaves every episode unreadable ("Parquet
# magic bytes not found"). All 455/456 of seed-75000's episodes were
# lost; the repo is abandoned. Replacement: airvla_v2_d3 (seed 77000,
# 120 pairs, target sized UNDER the wall so the writer closes).
# Sources are therefore d2 + d3; both jobs must end with
# DCOLLECT-EXIT=0 (a clean close is now a merge precondition).
D_MAN, D2_MAN, OUT_SPLIT, OUT_MAN = sys.argv[1:5]
SPLIT_SEED = 424245                      # D-episode split only (424244
                                         # is taken: C-wrapper's
                                         # rebalance sampling); the
                                         # v21 split is reused
                                         # verbatim, never redrawn
V21_EPS = 910


def banked_rows(path):
    rows = [json.loads(line) for line in open(path)]
    werr = {r["attempt"] for r in rows
            if r.get("reason") == "writer-error"}
    return [r for r in rows if r.get("banked")
            and r["attempt"] not in werr]


d1_rows = banked_rows(D_MAN)
d2_rows = banked_rows(D2_MAN)
d_rows = d1_rows + d2_rows              # must match aggregation order
print("d banked: %d + %d = %d" % (len(d1_rows), len(d2_rows),
                                  len(d_rows)), flush=True)

aggregate_datasets(["hanapasta/airvla_v21", "hanapasta/airvla_v2_d2",
                    "hanapasta/airvla_v2_d3"],
                   "hanapasta/airvla_v22")
ds = LeRobotDataset("hanapasta/airvla_v22")
total = ds.meta.total_episodes
assert total == V21_EPS + len(d_rows), (total, len(d_rows))
print("aggregated:", total, "episodes,", ds.meta.total_frames,
      "frames", flush=True)

prev = [json.loads(line) for line in
        open("e3_manifest_combined.jsonl")]
assert len(prev) == V21_EPS, len(prev)
with open(OUT_MAN, "w") as fh:
    for i, r in enumerate(prev + d_rows):
        fh.write(json.dumps(dict(r, combined_episode_index=i)) + "\n")

old = json.load(open("e3_split.json"))
rng = np.random.default_rng(SPLIT_SEED)
new_idx = list(range(V21_EPS, total))
units = {}
for i, r in zip(new_idx, d_rows):
    assert r["kind"] in ("pairA", "pairB"), r["kind"]
    # pair_id restarts at 0 in each collection job -- key units by
    # (collector_seed, pair_id) or pairs from the two jobs would fuse
    units.setdefault((r["collector_seed"], r["pair_id"]),
                     []).append(i)
pair_units = [tuple(v) for v in units.values()]


def split20(items):
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, round(0.2 * len(items)))
    return items[n_val:], items[:n_val]


p_tr_u, p_va_u = split20(pair_units)
p_tr = [i for u in p_tr_u for i in u]
p_va = [i for u in p_va_u for i in u]
train = sorted(old["train"] + p_tr)
val = sorted(old["val"] + p_va)
assert not set(train) & set(val)
assert len(train) + len(val) == total
json.dump(dict(train=train, val=val, seed=SPLIT_SEED,
               note="airvla_v21 split verbatim; D pairs split as "
                    "units (members share a side)"),
          open(OUT_SPLIT, "w"), indent=1)
print("split: train %d val %d (new pair-units %d/%d)"
      % (len(train), len(val), len(p_tr_u), len(p_va_u)), flush=True)

api = HfApi()
for repo in ("hanapasta/airvla_v2_d2", "hanapasta/airvla_v2_d3",
             "hanapasta/airvla_v22"):
    LeRobotDataset(repo).push_to_hub()
    info = api.dataset_info(repo)
    print("PUSH-READBACK %s revision %s files-ok" % (repo,
                                                     info.sha[:12]),
          flush=True)
print("MERGE-D-DONE", flush=True)
