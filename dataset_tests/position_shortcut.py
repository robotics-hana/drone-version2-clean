"""position_shortcut.py -- Test 2: is the target identifiable from position
alone?

The language prompt is behaviourally inert at 60k (decision log D50):
selection follows position regardless of the noun. This test measures
whether the DATASET made language redundant -- if any position-style
shortcut identifies the target reliably, the model never needed to read
the prompt during training, and the inert noun is a dataset-design
consequence rather than a model failure.

Reads frame 0 of each pick episode only. scene_state layout (verified
against collect_airvla.Runner.scene_state): [0:7] task-object pose,
[7:14] distractor pose. Drone start = observation.state[0:2].

Shortcut rules scored:
  position   : target = object closer to the mean target xy (in-sample)
  near-drone : target = object nearer the drone's start xy
  identity   : majority-class rule on which object is the target
               (equivalently "the larger object" -- the penguin is the
               physically larger of the two)

Decision rule (pre-registered, from the brief, before any numbers):
  any shortcut > 70% accurate => language is redundant in the training
  data; fix = re-collect manipulation episodes with the target location
  randomised. All shortcuts near 50% => the language was necessary and
  the model failed to learn it (points at vocabulary/training, not the
  collector).

usage: python position_shortcut.py <dataset_dir> <out_scatter.png>
"""
import glob
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(f, columns=[
    "episode_index", "task_index", "frame_index", "observation.state",
    "scene_state"]) for f in files)
f0 = df[(df.frame_index == 0) & df.task_index.isin([0, 1])]

tgt = np.stack(f0["scene_state"].to_numpy())[:, 0:2]
dis = np.stack(f0["scene_state"].to_numpy())[:, 7:9]
drone = np.stack(f0["observation.state"].to_numpy())[:, 0:2]
is_weight = (f0.task_index == 0).to_numpy()
n = len(f0)
print("POSITION SHORTCUT  n=%d pick episodes (frame 0)" % n)
print("decision rule (pre-registered): any shortcut >70%% accurate => "
      "language redundant in the data => re-collect with target location "
      "randomised; all near 50%% => model failure, not collector")

print("-- spawn statistics (m)")
print("%-11s %8s %8s %8s %8s %8s %8s" % ("", "x mean", "x std", "x range",
                                         "y mean", "y std", "y range"))
for name, a in (("target", tgt), ("distractor", dis)):
    print("%-11s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
          % (name, a[:, 0].mean(), a[:, 0].std(), np.ptp(a[:, 0]),
             a[:, 1].mean(), a[:, 1].std(), np.ptp(a[:, 1])))

# bounding-box overlap, each direction
def box_frac(a, b):
    lo, hi = b.min(0), b.max(0)
    return float(np.mean(np.all((a >= lo) & (a <= hi), axis=1)))
print("-- overlap: %.1f%% of target spawns inside the distractor's box; "
      "%.1f%% of distractor spawns inside the target's box"
      % (100 * box_frac(tgt, dis), 100 * box_frac(dis, tgt)))

# shortcut 1: closer to mean target position
mu = tgt.mean(0)
acc_pos = np.mean(np.linalg.norm(tgt - mu, axis=1)
                  < np.linalg.norm(dis - mu, axis=1))
# shortcut 2: nearer the drone start
acc_drone = np.mean(np.linalg.norm(tgt - drone, axis=1)
                    < np.linalg.norm(dis - drone, axis=1))
# shortcut 3: identity majority (= "the larger object" if that is penguin)
n_pen = int((~is_weight).sum())
acc_id = max(n_pen, n - n_pen) / n
print("-- shortcut accuracies")
print("   position (closer to mean target xy, in-sample): %5.1f%%"
      % (100 * acc_pos))
print("   nearer to drone start:                          %5.1f%%"
      % (100 * acc_drone))
print("   identity balance: weight %d / penguin %d -> majority rule %5.1f%%"
      % (n - n_pen, n_pen, 100 * acc_id))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(tgt[:, 0], tgt[:, 1], s=10, c="#b3423a", label="target spawn")
ax.scatter(dis[:, 0], dis[:, 1], s=10, c="#2e5e8f", label="distractor spawn")
ax.scatter(drone[:, 0], drone[:, 1], s=6, c="#999", alpha=0.5,
           label="drone start")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Frame-0 spawn positions, %d pick episodes" % n)
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(out, dpi=130)
print("plot ->", out)
