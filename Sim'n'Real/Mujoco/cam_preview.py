"""Preview the onboard cameras without running a whole episode.

Poses the drone mid-task (hovering over the workspace with the arm at the grasp
pose, holding a block) and dumps one PNG per camera. Edit the <camera> lines in
SkyGrip_full.xml, re-run this, look at the PNGs, repeat.

    python cam_preview.py            # all cameras, at the grasp
    python cam_preview.py approach   # at the approach hover instead

Writes cam_preview_<name>.png next to this script.
"""
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio
import collect_demos as C

where = sys.argv[1] if len(sys.argv) > 1 else "grasp"

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml")
d = mujoco.MjData(m)
ik = C.ArmIK(m)
rng = np.random.default_rng(5)

scene = C.randomise_episode(m, d, rng, "purple", "cylinder", "blue", "block")
q, off = ik.solve_drop(0.19)
obj = scene["target"]

if where == "approach":
    d.qpos[0:3] = obj + np.array([0.0, 0.10, 0.42]) - off
else:  # at the grasp: jaws on the object
    grasp_pt = obj + np.array([0.0, C.GRASP_Y_OFFSET,
                               scene["target_half_h"] - 0.010])
    d.qpos[0:3] = grasp_pt - off
d.qpos[3:7] = [1, 0, 0, 0]
d.qpos[7:9] = q
mujoco.mj_forward(m, d)

rend = mujoco.Renderer(m, height=C.IMG_H, width=C.IMG_W)
for key, cam in C.CAMERAS.items():
    rend.update_scene(d, camera=cam)
    out = f"cam_preview_{key}_{cam}.png"
    imageio.imwrite(out, rend.render())
    print(f"wrote {out}   ({cam}, posed at {where})")
