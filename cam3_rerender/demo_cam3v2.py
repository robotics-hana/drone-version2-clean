"""demo_cam3v2.py -- side-by-side demo of the CURRENT external camera vs a
candidate task-framed replacement, rendered over logged episode states.

Framing demo only, produced BEFORE any cluster work (user requirement).
Honest limits, stated on every frame: two per-episode quantities are NOT in
the logs and are approximated here -- the table's lateral offset (u_off,
drawn from U(+-0.10..0.28) at collection; rendered at 0) and the two bins'
random wood tints (rendered at XML defaults). The pre-flight report covers
how the real build recovers them.

Camera candidates are MuJoCo free cameras -- no model XML is created or
edited at this stage.

usage: python demo_cam3v2.py <states.json> <out.mp4> [candidate]
"""
import json
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

SIM = r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco"
RES = 512                      # stored dataset resolution
BODY = {"weight": "pick_weight", "plush penguin": "penguin"}

# Candidate poses: lookat is the table front-edge zone where every grasp
# happens; positions sit on the stairs side of the room (-x, +y quadrant),
# far enough back to keep the bin spawn box (x 1.1..1.7, y 1.2..2.0) and the
# whole table (0.9 x 0.6 m) in frame.
CANDS = {
    "A": dict(pos=[-1.55, 2.05, 1.75], look=[0.35, 0.85, 0.35], fovy=58),
    "B": dict(pos=[-1.10, 1.65, 1.35], look=[0.40, 0.90, 0.35], fovy=62),
    "C": dict(pos=[-1.90, 2.50, 2.10], look=[0.30, 0.85, 0.35], fovy=52),
}

st = json.load(open(sys.argv[1]))
out = sys.argv[2]
cand = CANDS[sys.argv[3] if len(sys.argv) > 3 else "B"]

m = mujoco.MjModel.from_xml_path(SIM + "/SkyGrip_airvla.xml")
d = mujoco.MjData(m)
rend = mujoco.Renderer(m, height=RES, width=RES)
j1 = m.joint("Joint_1").qposadr[0]
j2 = m.joint("Joint_2").qposadr[0]
gadr = m.joint("right_clamp").qposadr[0]
adr = {k: m.jnt_qposadr[m.body_jntadr[m.body(v).id]] for k, v in BODY.items()}
table = m.body("table").id
bin_id = m.body("bin").id
gate = m.body("gate").id

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
p, look = np.array(cand["pos"]), np.array(cand["look"])
v = look - p
cam.lookat[:] = look
cam.distance = float(np.linalg.norm(v))
cam.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
# MuJoCo elevation: negative = camera above lookat, looking down.
cam.elevation = float(np.degrees(np.arcsin(v[2] / cam.distance)))
m.vis.global_.fovy = cand["fovy"]

w = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                       macro_block_size=1)
for e in st[:2]:                                   # one weight, one penguin
    tgt = "plush penguin" if "penguin" in e["task"] else "weight"
    oth = "weight" if tgt == "plush penguin" else "plush penguin"
    S = np.array(e["state"])
    SC = np.array(e["scene"])
    # scene placement from LOGGED values; approximations flagged on-frame
    txy = SC[0, 0:2]
    m.body_pos[table][0:2] = [float(np.clip(txy[0], -0.55, 0.55)),
                              txy[1] - 0.21]        # u_off approximated as 0
    m.body_pos[table][2] = 0.04
    m.body_pos[bin_id][0:2] = SC[0, 14:16]
    m.body_pos[bin_id][2] = 0.04
    m.body_pos[gate][2] = -3.0                      # picks: gate parked
    m.body_pos[m.body("stand").id][2] = -3.0
    m.body_pos[m.body("stand2").id][2] = -3.0
    for k in range(0, len(S), 3):
        d.qpos[0:7] = S[k, 0:7]
        d.qpos[j1], d.qpos[j2] = S[k, 8], S[k, 9]
        d.qpos[gadr] = S[k, 7] * 0.016
        d.qpos[adr[tgt]:adr[tgt] + 7] = SC[k, 0:7]
        d.qpos[adr[oth]:adr[oth] + 7] = SC[k, 7:14]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera="lab_external")
        old = rend.render().copy()
        rend.update_scene(d, camera=cam)
        new = rend.render().copy()
        img = np.concatenate([old, new], axis=1)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.text((8, 8), "CURRENT camera3 (lab_external)", fill=(255, 255, 0))
        dr.text((RES + 8, 8), "CANDIDATE %s  pos(%.2f,%.2f,%.2f) fovy %d"
                % (sys.argv[3] if len(sys.argv) > 3 else "B",
                   *cand["pos"], cand["fovy"]), fill=(255, 255, 0))
        dr.text((RES + 8, 26),
                "demo: table u_off approximated 0; bin tint = default",
                fill=(255, 160, 80))
        dr.text((8, RES - 22), "ep%d %s  frame %d" % (e["ep"], tgt, k),
                fill=(0, 255, 255))
        w.append_data(np.asarray(im))
w.close()
print("wrote", out)
