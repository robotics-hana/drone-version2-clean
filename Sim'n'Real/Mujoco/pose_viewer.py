"""pose_viewer.py -- open a SkyGrip model in the interactive MuJoCo
viewer with gravity DISABLED, so the drone floats motionless in a
preset pose instead of falling (the stock viewer cannot start
paused). Use the right panel's Joint sliders to pose Joint_1,
Joint_2 and the clamps; left-drag to orbit, scroll to zoom; F7 for
a screenshot. Re-enable gravity live via the viewer's Physics
panel (Option > gravity) if you ever want dynamics back.

usage:  python pose_viewer.py [model.xml]
        (default SkyGrip_full.xml; use SkyGrip_airvla.xml for the
        full lab scene)
"""
import sys

import mujoco
import mujoco.viewer

XML = sys.argv[1] if len(sys.argv) > 1 else "SkyGrip_full.xml"
m = mujoco.MjModel.from_xml_path(XML)
m.opt.gravity[:] = 0                     # float: nothing moves
m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
m.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
d = mujoco.MjData(m)

# preset: the deployed pose from the figures -- arm straight down,
# wrist rotated so the open claws point down, drone at 2 m
d.qpos[0:3] = [0, 0, 2.0]
d.qpos[3:7] = [1, 0, 0, 0]
for name, val in (("Joint_1", 0.0), ("Joint_2", 1.57),
                  ("right_clamp", 0.016), ("left_clamp", -0.016)):
    try:
        d.qpos[m.joint(name).qposadr[0]] = val
    except KeyError:
        pass
# park any loose objects far away so only the drone is in frame
for i in range(m.njnt):
    name = m.joint(i).name
    if name.endswith("_free_joint") and name != "drone_free_joint" \
            or name in ("penguin_joint", "pick_weight_joint",
                        "mustard_free_joint"):
        a = m.joint(i).qposadr[0]
        d.qpos[a:a + 3] = [5, 5, 0.2]
mujoco.mj_forward(m, d)

print("Gravity OFF -- the drone floats. Pose with the Joint sliders "
      "(right panel), orbit with the mouse, F7 = screenshot.")
mujoco.viewer.launch(m, d)
