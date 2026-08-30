"""camera_pixels.py -- how many pixels does the TARGET occupy in each of the
policy's three camera streams, at the 224 px input resolution, as a function
of drone-to-object distance?

These numbers underpin the observation-deficit analysis (decision log D47):
camera3 (the base_0_rgb slot) carries almost nothing; camera2 (nose) is thin
at commitment range and rich late; camera1 (wrist) resolves only close in.

Measurement, not simulation: frames are reconstructed from logged expert
states (states.json: observation.state + scene_state for 4 training
episodes), rendered with segmentation, and target-geom pixels counted.
Run locally:  python camera_pixels.py <states.json> | tee camera_pixels.log
"""
import json
import sys

import numpy as np
import mujoco

SIM = r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco"
RES = 224
BODY = {"weight": "pick_weight", "plush penguin": "penguin"}

st = json.load(open(sys.argv[1]))
m = mujoco.MjModel.from_xml_path(SIM + "/SkyGrip_airvla.xml")
d = mujoco.MjData(m)
rend = mujoco.Renderer(m, height=RES, width=RES)
rend.enable_segmentation_rendering()
j1 = m.joint("Joint_1").qposadr[0]
j2 = m.joint("Joint_2").qposadr[0]
gadr = m.joint("right_clamp").qposadr[0]
adr = {k: m.jnt_qposadr[m.body_jntadr[m.body(v).id]] for k, v in BODY.items()}


def subtree_geoms(name):
    b = m.body(name).id
    return set(g for g in range(m.ngeom)
               if m.body_rootid[m.geom_bodyid[g]] == b)


rows = []                       # (distance, cam3_px, cam2_px, cam1_px)
for e in st:
    tgt = "plush penguin" if "penguin" in e["task"] else "weight"
    oth = "weight" if tgt == "plush penguin" else "plush penguin"
    tg = subtree_geoms(BODY[tgt])
    S = np.array(e["state"])
    SC = np.array(e["scene"])
    for k in range(0, len(S), 8):
        d.qpos[0:7] = S[k, 0:7]
        d.qpos[j1], d.qpos[j2] = S[k, 8], S[k, 9]
        d.qpos[gadr] = S[k, 7] * 0.016
        d.qpos[adr[tgt]:adr[tgt] + 7] = SC[k, 0:7]
        d.qpos[adr[oth]:adr[oth] + 7] = SC[k, 7:14]
        mujoco.mj_forward(m, d)
        dist = float(np.linalg.norm(S[k, 0:2] - SC[k, 0:2]))
        px = []
        for cam in ("lab_external", "scene_cam", "wrist_cam"):
            rend.update_scene(d, camera=cam)
            seg = rend.render()
            vis = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
            px.append(int(np.isin(seg[:, :, 0], list(tg))[vis].sum()))
        rows.append((dist, *px))

print("target pixels at 224 px, %d frames from %d expert episodes"
      % (len(rows), len(st)))
print("%-14s %6s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s"
      % ("distance bin", "n", "c3 min", "c3 med", "c3 max",
         "c2 min", "c2 med", "c2 max", "c1 min", "c1 med", "c1 max"))
bins = [(1.0, 9.9), (0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
for lo, hi in bins:
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        continue
    a = np.array(sel)
    out = ["%.1f-%.1f m" % (lo, hi), "%d" % len(sel)]
    for c in (1, 2, 3):
        out += ["%d" % a[:, c].min(), "%.0f" % np.median(a[:, c]),
                "%d" % a[:, c].max()]
    print("%-14s %6s | %8s %8s %8s | %8s %8s %8s | %8s %8s %8s" % tuple(out))
print("columns: c3 = lab_external (base_0_rgb slot), c2 = scene_cam (nose),")
print("         c1 = wrist_cam. Frames from EXPERT approaches; a policy that")
print("         flies elsewhere may see less.")
