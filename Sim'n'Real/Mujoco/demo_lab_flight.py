"""demo_lab_flight.py -- presentation demo: the drone flying inside the
scanned-lab environment, filmed by a slow orbiting camera.

Replays logged expert episodes (drone pose, arm, gripper, both objects,
straight from the dataset tables -- no policy, no expert re-run) inside
SkyGrip_airvla.xml, whose room is the lab capture baked to textured
meshes (lab.xml, generated from the scan by prepare_lab_scene.py). The
orbit camera circles the workspace so the scanned room, not just the
task, is in view. Renders one pick-and-place and one gate flight.

usage: python demo_lab_flight.py <dataset_dir> <out.mp4>
"""
import glob
import sys

import numpy as np
import pandas as pd
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

W, H = 960, 720
BODY = {"weight": "pick_weight", "plush penguin": "penguin"}

ds, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(ds + "/data/**/*.parquet", recursive=True))
df = pd.concat(pd.read_parquet(f, columns=[
    "episode_index", "task_index", "frame_index", "observation.state",
    "scene_state"]) for f in files)

spec = mujoco.MjSpec.from_file("SkyGrip_airvla.xml")
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
m = spec.compile()
d = mujoco.MjData(m)
rend = mujoco.Renderer(m, height=H, width=W)
j1 = m.joint("Joint_1").qposadr[0]
j2 = m.joint("Joint_2").qposadr[0]
gadr = m.joint("right_clamp").qposadr[0]
adr = {k: m.jnt_qposadr[m.body_jntadr[m.body(v).id]] for k, v in BODY.items()}
table, bin_id, gate = m.body("table").id, m.body("bin").id, m.body("gate").id

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# one pick (task 0 = weight) and one nav (task 2 = gate + weight)
EPS = [(int(df[df.task_index == 0].episode_index.iloc[0]), "pick",
        "pick up the weight and put it in the wooden box"),
       (int(df[df.task_index == 2].episode_index.iloc[0]), "nav",
        "fly through the gate and hover over the weight")]

w = imageio.get_writer(out, fps=20, codec="libx264", quality=8,
                       macro_block_size=1)
for ep, kind, task in EPS:
    g = df[df.episode_index == ep].sort_values("frame_index")
    S = np.stack(g["observation.state"].to_numpy())
    SC = np.stack(g["scene_state"].to_numpy())
    tgt, oth = "weight", "plush penguin"
    txy = SC[0, 0:2]
    m.body_pos[table][0:2] = [float(np.clip(txy[0], -0.55, 0.55)),
                              txy[1] - 0.21]
    m.body_pos[table][2] = 0.04
    m.body_pos[bin_id][0:2] = SC[0, 14:16]
    m.body_pos[bin_id][2] = 0.04
    if kind == "nav":
        m.body_pos[gate][0:2] = [0.7, -0.6]
        m.body_pos[gate][2] = 0.04
    else:
        m.body_pos[gate][2] = -3.0
    # retired scenery: pedestals underground, mustard bottle off-stage
    # (reset_scene does this; direct qpos replay must repeat it)
    for st in ("stand", "stand2"):
        m.body_pos[m.body(st).id][2] = -3.0
    ma = m.jnt_qposadr[m.body_jntadr[m.body("mustard_bottle").id]]
    d.qpos[ma:ma + 3] = [-0.95, 2.45, 0.0]
    n = len(S)
    for k in range(0, n, 2):
        d.qpos[0:7] = S[k, 0:7]
        d.qpos[j1], d.qpos[j2] = S[k, 8], S[k, 9]
        d.qpos[gadr] = S[k, 7] * 0.016
        d.qpos[adr[tgt]:adr[tgt] + 7] = SC[k, 0:7]
        d.qpos[adr[oth]:adr[oth] + 7] = SC[k, 7:14]
        mujoco.mj_forward(m, d)
        # orbit: one slow revolution per episode, lookat drifting between
        # the workspace centre and the drone so both stay framed
        t = k / max(1, n - 1)
        centre = 0.55 * np.array([0.1, 0.5, 0.5]) + 0.45 * d.qpos[0:3]
        cam.lookat[:] = centre
        cam.distance = 3.1
        cam.azimuth = -90 + 360 * t
        cam.elevation = -18
        rend.update_scene(d, camera=cam)
        im = Image.fromarray(rend.render().copy())
        dr = ImageDraw.Draw(im)
        dr.text((10, 8), task, fill=(255, 255, 0))
        dr.text((10, H - 22),
                "expert replay ep%d | scanned-lab environment (lab.xml, "
                "from the 3DGS capture) | 2x speed" % ep,
                fill=(0, 255, 255))
        w.append_data(np.asarray(im))
w.close()
print("wrote", out)
