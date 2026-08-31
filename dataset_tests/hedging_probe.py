"""hedging_probe.py -- Test A: is the policy hedging between the two
candidate objects?

If the policy cannot resolve WHICH object the prompt names, it should fly
toward something like the midpoint of the two candidates rather than to
either one. Half the trained inter-object separation is 225-310 mm and the
frozen-baseline median miss is 242 mm -- suspiciously close.

The distractor's position is not in any log, but it is a deterministic
function of the harness RNG stream: this script replays Runner(77000) and
the frozen run's exact sampling sequence (bxy draw, --startfix rejection
loop, reset_scene) for the 60 pick episodes of the frozen n=60 baseline,
recovering every scene's distractor xy. Reconstruction is validated
against the logged end-of-episode target xy (which equals the spawn for
every episode where the object was never moved).

Per episode: distance from the FINAL gripper position (last logged jaws
xy in eval_traj_v2.jsonl, tag frozen60k_n60) to target, distractor, and
their midpoint. Supplementary: the final jaws position projected onto the
target->distractor axis (0 = target, 0.5 = midpoint, 1 = distractor).

Decision rule (pre-registered, from the brief, before any numbers):
  if the midpoint is the closest of the three in MORE THAN HALF the
  episodes, selection failure is confirmed as the dominant lateral error
  and the v2 design must attack it directly.

usage: python hedging_probe.py <eval_traj_v2.jsonl>
(run from Sim'n'Real/Mujoco so collect_airvla imports)
"""
import json
import sys

import numpy as np

import collect_airvla as A

TRAJ = sys.argv[1]

recs = [json.loads(l) for l in open(TRAJ, encoding="utf-8", errors="ignore")
        if '"frozen60k_n60"' in l]
assert len(recs) == 60, "expected 60 frozen-run pick records, got %d" % len(recs)

r = A.Runner(77000)
rng = r.rng
scenes = []
for i in range(60):
    obj = "plush penguin" if i % 2 == 0 else "weight"
    # exact replica of the frozen harness sampling (eval_pi0.run_episode,
    # --startfix path): same draws, same order, same rejection loop
    bxy = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.20, 1.00)])
    for _ in range(300):
        start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.6),
                          rng.uniform(0.21, 0.30)])
        if (float(np.linalg.norm(start[0:2] - bxy)) >= 0.70
                and start[1] >= bxy[1] + 0.35):
            break
    r.reset_scene(bxy, start, obj=obj)
    other = [o for k, o in r.objs.items() if k != obj][0]
    scenes.append(dict(obj=obj, tgt=bxy.copy(), bin=r.bin_xy.copy(),
                       dis=np.array(r.data.qpos[other["adr"]:
                                                other["adr"] + 2])))

# validation: the bin is sampled at reset and physically immovable, so
# the logged bin_xy must reproduce exactly if the stream is faithful.
# (The target's logged end-of-episode xy is NOT a validator: objects get
# physically disturbed during episodes -- measured below.)
err = np.array([np.linalg.norm(np.array(rec["bin_xy"]) - s["bin"])
                for rec, s in zip(recs, scenes)])
ok = int((err < 0.001).sum())
print("reconstruction check: %d/60 bin positions exact to <1 mm "
      "(max %.4f m)" % (ok, err.max()))
assert ok == 60, "scene stream reconstruction unfaithful -- do not proceed"
moved = np.array([np.linalg.norm(np.array(rec["obj_xy"]) - s["tgt"])
                  for rec, s in zip(recs, scenes)])
print("secondary: task object displaced >20 mm from spawn by episode end "
      "in %d/60 episodes (max %.2f m) -- the drone disturbs the scene"
      % (int((moved > 0.02).sum()), moved.max()))

d_t, d_d, d_m, axis_pos = [], [], [], []
for rec, s in zip(recs, scenes):
    jaws = np.array(rec["traj"][-1][4:6])
    mid = 0.5 * (s["tgt"] + s["dis"])
    d_t.append(np.linalg.norm(jaws - s["tgt"]))
    d_d.append(np.linalg.norm(jaws - s["dis"]))
    d_m.append(np.linalg.norm(jaws - mid))
    u = s["dis"] - s["tgt"]
    axis_pos.append(float((jaws - s["tgt"]) @ u / (u @ u)))
d_t, d_d, d_m = map(np.array, (d_t, d_d, d_m))
axis_pos = np.array(axis_pos)

print("decision rule (pre-registered): midpoint closest in >1/2 of "
      "episodes => selection failure confirmed as the dominant lateral "
      "error; v2 must attack it directly")
print("medians (mm): target %.0f | midpoint %.0f | distractor %.0f"
      % (1000 * np.median(d_t), 1000 * np.median(d_m),
         1000 * np.median(d_d)))
w = np.argmin(np.stack([d_t, d_m, d_d]), axis=0)
print("closest of the three: target %d/60 | midpoint %d/60 | "
      "distractor %d/60" % ((w == 0).sum(), (w == 1).sum(), (w == 2).sum()))
print("final position on target->distractor axis "
      "(0=target, 0.5=midpoint, 1=distractor):")
print("  median %.2f  IQR [%.2f, %.2f]"
      % (np.median(axis_pos), np.quantile(axis_pos, .25),
         np.quantile(axis_pos, .75)))
for obj in ("weight", "plush penguin"):
    m = np.array([s["obj"] == obj for s in scenes])
    print("  %s targets (n=%d): axis median %.2f, midpoint-closest %d"
          % (obj, m.sum(), np.median(axis_pos[m]), (w[m] == 1).sum()))
