"""reversal_probe.py -- Test 1: does the expert reverse before grasping?

A non-monotonic final approach is hard for a policy that predicts fifty
actions and executes them open-loop, so whether the demonstrations contain
commanded reversals bears directly on the residual pick failure.

Reads the dataset tables only (no GPU, no simulation, no training). The
dataset itself is read-only; this script consumes a copied-out snapshot of
the parquet tables.

Method per pick episode (task_index 0 or 1):
  1. grasp tick = final open->closed crossing of the commanded grip
     channel (action[6] < 0.9, the collector's own quality threshold)
  2. window = the 60 ticks (6 s at 10 Hz) before the grasp tick
  3. approach direction = unit vector, drone xy at window start -> task
     object xy at grasp tick (scene_state[0:2])
  4. project each tick's commanded (dx, dy) = action[0:2] onto it
  5. deadband 1 mm, then count sign changes
  6. cumulative projected displacement; largest backward excursion =
     max over t of (running peak - cumulative[t])

Run twice: on COMMANDED actions (the labels the policy learns -- the one
that matters) and on MEASURED body position deltas (observation.state[0:2],
where a reversal could just be overshoot-and-settle).

Decision rule (pre-registered, from the brief, before any numbers):
  if >= 40% of pick episodes contain a commanded backward excursion
  > 2 cm, the reversal is a real feature of the training data; next step
  is the cheap test (filter to non-reversal episodes, retrain, evaluate)
  rather than a collection campaign. Report the survivor count.

usage: python reversal_probe.py <dataset_dir> <out_plot.png>
"""
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEADBAND = 0.001                 # 1 mm per tick
WINDOW = 60                      # ticks before the grasp
EXCURSION_BAR = 0.02             # 2 cm decision threshold
GRIP_CLOSED = 0.9                # collector's quality-harness threshold

ds, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(f, columns=[
    "episode_index", "task_index", "action", "observation.state",
    "scene_state"]) for f in files)

rows, curves, skipped = [], [], 0
for ep, g in df[df.task_index.isin([0, 1])].groupby("episode_index"):
    act = np.stack(g["action"].to_numpy())
    st = np.stack(g["observation.state"].to_numpy())
    sc = np.stack(g["scene_state"].to_numpy())
    grip = act[:, 6]
    cross = np.where((grip[1:] < GRIP_CLOSED) & (grip[:-1] >= GRIP_CLOSED))[0] + 1
    if len(cross) == 0 or cross[-1] < 20:
        skipped += 1
        continue
    tg = int(cross[-1])
    w0 = max(0, tg - WINDOW)
    v = sc[tg, 0:2] - st[w0, 0:2]           # approach direction
    n = np.linalg.norm(v)
    if n < 1e-6:
        skipped += 1
        continue
    u = v / n
    res = {}
    for name, dxy in (("cmd", act[w0:tg, 0:2]),
                      ("pos", np.diff(st[w0:tg + 1, 0:2], axis=0))):
        proj = dxy @ u
        cum = np.cumsum(proj)
        exc = float(np.max(np.maximum.accumulate(cum) - cum))
        live = proj[np.abs(proj) >= DEADBAND]
        signs = np.sign(live)
        flips = int(np.sum(signs[1:] != signs[:-1])) if len(live) > 1 else 0
        res[name + "_flips"] = flips
        res[name + "_exc"] = exc
        if name == "cmd":
            curves.append(cum)
    rows.append(dict(ep=int(ep), obj="weight" if g.task_index.iloc[0] == 0
                     else "penguin", **res))

r = pd.DataFrame(rows)
print("REVERSAL PROBE  n=%d pick episodes analysed, %d skipped "
      "(no grip crossing or too early)" % (len(r), skipped))
print("decision rule (pre-registered): >=40%% of episodes with commanded "
      "excursion > %.0f cm => reversal is real; then filter+retrain, "
      "not re-collect" % (EXCURSION_BAR * 100))
for name, label in (("cmd", "COMMANDED actions (the training labels)"),
                    ("pos", "measured body position (context only)")):
    e, f = r[name + "_exc"], r[name + "_flips"]
    print("-- %s" % label)
    print("   episodes with >=1 sign change after deadband: %d/%d (%.0f%%)"
          % ((f > 0).sum(), len(r), 100 * (f > 0).mean()))
    print("   episodes with backward excursion > 2 cm:      %d/%d (%.0f%%)"
          % ((e > EXCURSION_BAR).sum(), len(r),
             100 * (e > EXCURSION_BAR).mean()))
    print("   sign changes per episode: median %.0f  max %d"
          % (f.median(), f.max()))
    print("   backward excursion (m):   median %.4f  max %.4f"
          % (e.median(), e.max()))
for obj in ("weight", "penguin"):
    q = r[r.obj == obj]
    print("-- %s only (n=%d): cmd excursion>2cm %d (%.0f%%), "
          "median exc %.4f m, median flips %.0f"
          % (obj, len(q), (q.cmd_exc > EXCURSION_BAR).sum(),
             100 * (q.cmd_exc > EXCURSION_BAR).mean(),
             q.cmd_exc.median(), q.cmd_flips.median()))
surv = int((r.cmd_exc <= EXCURSION_BAR).sum())
print("survivors of a no-reversal filter: %d/%d episodes" % (surv, len(r)))

rng = np.random.default_rng(0)
pick = rng.choice(len(curves), size=min(20, len(curves)), replace=False)
fig, ax = plt.subplots(figsize=(8, 5))
for i in pick:
    ax.plot(curves[i], lw=1, alpha=0.7)
ax.set_xlabel("tick within final-approach window (10 Hz)")
ax.set_ylabel("cumulative commanded displacement\nalong approach (m)")
ax.set_title("Final 6 s before grasp: cumulative along-approach "
             "displacement, 20 random pick episodes")
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("plot ->", out)
