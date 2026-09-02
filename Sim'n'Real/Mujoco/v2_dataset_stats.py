"""v2_dataset_stats.py -- section-C dataset balance checks: the numbers
the dissertation checklist asks for, computed from the collection
manifest + dataset tables. Run on the trial and on the real dataset.

Reports: composition by flavour/object; success + rejection statistics;
spatial coverage table (min/max/mean/SD for target x/y, distractor x/y);
episode-length distribution; parking-window distribution; and the two
pre-registered gates (position shortcut <= 55%, parking window <= 1 tick
median) with PASS/FAIL verdicts.

usage: python v2_dataset_stats.py <manifest.jsonl> <dataset_root>
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

manifest, root = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(manifest)]
banked = [r for r in rows if r.get("banked")]
rej = [r for r in rows if not r.get("banked")]
print("=== composition ===")
for k in ("std", "corr", "nav"):
    ks = [r for r in banked if r["kind"] == k]
    objs = [r.get("obj", "?") for r in ks]
    print("  %-5s %3d banked  (weight %d / penguin %d)"
          % (k, len(ks), objs.count("weight"), objs.count("plush penguin")))
print("  rejected attempts: %d" % len(rej))
for r in rej[:10]:
    print("    slot %s [%s]: %s" % (r.get("slot"), r.get("kind"),
                                    r.get("reason")))

print("=== spatial coverage (banked, m) ===")
f = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(x, columns=[
    "episode_index", "frame_index", "scene_state"]) for x in f)
f0 = df[df.frame_index == 0]
sc = np.stack(f0["scene_state"].to_numpy())
tgt, dis = sc[:, 0:2], sc[:, 7:9]
print("%-14s %7s %7s %7s %7s" % ("variable", "min", "max", "mean", "SD"))
for name, a in (("target x", tgt[:, 0]), ("target y", tgt[:, 1]),
                ("distractor x", dis[:, 0]), ("distractor y", dis[:, 1])):
    print("%-14s %7.2f %7.2f %7.2f %7.2f"
          % (name, a.min(), a.max(), a.mean(), a.std()))
lens = df.groupby("episode_index").size()
print("%-14s %7d %7d %7.0f %7.0f" % ("episode ticks", lens.min(),
                                     lens.max(), lens.mean(), lens.std()))

print("=== pre-registered gates ===")
pk = [r.get("parking") for r in banked
      if r["kind"] != "nav" and r.get("parking") is not None]
med_pk = float(np.median(pk)) if pk else -1
print("  parking window: median %.0f tick(s), max %d  -> %s"
      % (med_pk, max(pk) if pk else -1,
         "PASS" if 0 <= med_pk <= 1 else "FAIL"))
# position shortcut on the collected data: nearest-centroid, honest CV
if len(tgt) >= 10:
    accs = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        hold = rng.choice(len(tgt), size=max(2, len(tgt) // 5),
                          replace=False)
        tr = np.setdiff1d(np.arange(len(tgt)), hold)
        mu = tgt[tr].mean(0)
        accs.append(np.mean(
            np.linalg.norm(tgt[hold] - mu, axis=1)
            < np.linalg.norm(dis[hold] - mu, axis=1)))
    acc = float(np.mean(accs))
    print("  position shortcut (CV nearest-centroid): %.1f%%  -> %s"
          % (100 * acc, "PASS" if acc <= 0.55 else "FAIL"))
print("=== done ===")
