"""parking_window.py -- Test C: characterise the expert's parking window
precisely.

The reversal probe showed the commanded horizontal motion is exactly zero
in the final 6 s before the grip closes. This test measures the exact
window: for every pick episode, the last tick at which the commanded
horizontal (and separately vertical) motion is non-zero, reported as
ticks before the grip close. This is the number the v2 expert has to
change, and the figure to quote in the write-up.

No decision rule -- this is a measurement to parameterise change 1 of the
v2 collection (continuous terminal approach).

usage: python parking_window.py <dataset_dir>
"""
import glob
import sys

import numpy as np
import pandas as pd

ds = sys.argv[1]
files = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(f, columns=[
    "episode_index", "task_index", "action"]) for f in files)

rows = []
for ep, g in df[df.task_index.isin([0, 1])].groupby("episode_index"):
    act = np.stack(g["action"].to_numpy())
    grip = act[:, 6]
    cr = np.where((grip[1:] < 0.9) & (grip[:-1] >= 0.9))[0] + 1
    if len(cr) == 0:
        continue
    tg = int(cr[-1])
    horiz = np.linalg.norm(act[:tg, 0:2], axis=1)
    vert = np.abs(act[:tg, 2])
    row = dict(ep=int(ep))
    for name, sig, eps in (("h_exact", horiz, 1e-9), ("h_1mm", horiz, 1e-3),
                           ("v_exact", vert, 1e-9), ("v_1mm", vert, 1e-3)):
        nz = np.where(sig > eps)[0]
        row[name] = tg - int(nz[-1]) if len(nz) else tg
    rows.append(row)

r = pd.DataFrame(rows)
print("PARKING WINDOW  n=%d pick episodes  (ticks before grip close of "
      "the last non-zero command; 10 ticks = 1 s)" % len(r))
print("%-28s %8s %8s %8s %8s" % ("channel / threshold", "median",
                                 "IQR lo", "IQR hi", "max"))
for name, label in (("h_exact", "horizontal, any motion"),
                    ("h_1mm", "horizontal, > 1 mm/tick"),
                    ("v_exact", "vertical, any motion"),
                    ("v_1mm", "vertical, > 1 mm/tick")):
    print("%-28s %8.0f %8.0f %8.0f %8.0f"
          % (label, r[name].median(), r[name].quantile(.25),
             r[name].quantile(.75), r[name].max()))
print("min horizontal window across all episodes: %d ticks (%.1f s)"
      % (r.h_exact.min(), r.h_exact.min() / 10))
