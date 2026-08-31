"""terminal_profile.py -- v2 sign-off artefact 3: the current vs proposed
terminal approach, as commanded along-approach speed against ticks before
grip close.

Current profile: measured from the dataset (median + IQR across all 225
pick episodes, aligned to the grip-close tick). Proposed v2 expert law:
proportional closure with a floor and close-in-motion --

    v(t) = clip(0.35 * d(t) / FPS, 3 mm/tick, 30 mm/tick)
    grip fires at d <= 15 mm, WITHOUT stopping first

so commanded motion is non-zero at every tick through the close. The v1
window this must beat: zero commanded horizontal motion for the final
104-117 ticks (parking_window.py).

usage: python terminal_profile.py <dataset_dir> <out.png>
"""
import glob, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(f, columns=["episode_index","task_index","action"]) for f in files)
W = 300
prof = []
for ep, g in df[df.task_index.isin([0,1])].groupby("episode_index"):
    act = np.stack(g["action"].to_numpy())
    grip = act[:,6]
    cr = np.where((grip[1:] < 0.9) & (grip[:-1] >= 0.9))[0] + 1
    if len(cr)==0 or cr[-1] < W: continue
    tg = int(cr[-1])
    prof.append(np.linalg.norm(act[tg-W:tg,0:2],axis=1))
P = np.stack(prof)*1000
med, lo, hi = np.median(P,0), np.quantile(P,.25,0), np.quantile(P,.75,0)

# proposed law simulated from d0 = 1.0 m
d, v2 = 1.0, []
while d > 0.015:
    v = float(np.clip(0.35*d/10, 0.003, 0.030)); v2.append(v*1000); d -= v
t2 = np.arange(-len(v2),0)

fig, ax = plt.subplots(figsize=(9,4.6))
t = np.arange(-W,0)
ax.fill_between(t, lo, hi, alpha=.25, color="#b3423a", lw=0)
ax.plot(t, med, color="#b3423a", lw=1.8, label="v1 expert (measured, median + IQR, n=%d)" % len(P))
ax.plot(t2, v2, color="#2e7d4f", lw=1.8, label="v2 proposed law (close fires in motion at 15 mm)")
ax.axvspan(-107, 0, color="k", alpha=.06)
ax.annotate("v1 parks: zero command\nfor final 107 ticks", xy=(-54,8), ha="center", fontsize=9)
ax.axvline(0, color="k", lw=.6)
ax.set_xlabel("ticks before grip close (10 Hz)")
ax.set_ylabel("commanded horizontal speed (mm/tick)")
ax.set_title("Terminal approach: v1 measured vs v2 proposed")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("episodes with >=300 pre-close ticks:", len(P))
print("v2 law: %d ticks from 1.0 m to close, min speed %.1f mm/tick, never zero" % (len(v2), min(v2)))
print("plot ->", out)
