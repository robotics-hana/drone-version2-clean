"""comp_setcheck.py -- C1 gate: chained expert replay on composite
scenes (pre-registered 2026-09-10: "harness validated FIRST by
chained expert replay -- expert nav phase + pick phase on the same
scene must complete the composite under the new scoring").

Runs the EXPERT's own stock stages (nav_v2 -> pick_v2 -> place_v2)
on comp scenes (nav spawn + pick-band objects, seed family 95000)
and applies the composite scoring. Pass = every episode crosses the
gate, grasps, and places inside the box. Exit 0 = gate pass.

usage (GPU node, pinned env): python comp_setcheck.py
"""
import sys

import numpy as np

import collect_airvla as A
import collect_v2 as V

SEED = 95000
N_EP = 6

COMP_PROMPT = ("fly through the gate and hover over the {obj}, "
               "then pick up the {obj} and put it in the wooden box")

r = V.V2Runner(SEED)
ok = True
for ep in range(N_EP):
    obj, start, alt, tgt = r.reset_scene_v2(comp=True)
    task = COMP_PROMPT.format(obj=obj)
    frames = []
    hover_ok = r.nav_v2(frames, task)
    grasped = r.pick_v2(frames, task, alt)
    d_bin = r.place_v2(frames, task) if grasped else float("nan")
    S = np.stack([f["observation.state"] for f in frames])
    gx = r._gate_x
    crossed = False
    for k in range(1, len(S)):
        if (S[k - 1, 1] < -0.6 <= S[k, 1]
                and abs(float(S[k, 0]) - gx) < 0.45
                and 0.38 < float(S[k, 2]) < 1.44):
            crossed = True
            break
    placed = bool(grasped and d_bin == d_bin and d_bin <= 0.15)
    success = bool(crossed and grasped and placed)
    if not success:
        ok = False
    print("EP%d obj=%-14s crossed=%-5s hover=%-5s grasped=%-5s "
          "d_bin=%s mm  gate_hits=%d table_hits=%d obj_hits=%d %s "
          "(%d ticks)"
          % (ep, obj, crossed, hover_ok, grasped,
             (round(1000 * d_bin, 1) if d_bin == d_bin else None),
             r._gate_hits, r._table_hits, r._obj_hits,
             "SUCCESS" if success else "FAIL", len(frames)),
          flush=True)

print("COMP-SETCHECK-%s" % ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
