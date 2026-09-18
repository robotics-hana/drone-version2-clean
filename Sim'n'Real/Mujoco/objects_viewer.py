"""objects_viewer.py -- open the MuJoCo viewer with the blue plush
penguin and the calibration weight sitting on the task table.

usage:  python objects_viewer.py

Everything else (drone, bottle, clutter) is parked out of sight.
The objects are dropped onto the table with real physics for half a
second so they rest naturally, then gravity is switched off so the
scene holds still while you orbit.  Left-drag = orbit, right-drag =
pan, scroll = zoom, F7 = screenshot.
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("SkyGrip_airvla.xml")
m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
m.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
d = mujoco.MjData(m)

# NOTE: the TRAINING penguin is near-black (rgba 0.06 0.06 0.07 in
# the training XML, verified identical on the cluster) -- "blue
# penguin" in the docs is its historical label. Set BLUE_PENGUIN
# to False to see the exact training appearance.
BLUE_PENGUIN = True
if BLUE_PENGUIN:
    for gname in ("penguin_base", "penguin_body", "penguin_head"):
        m.geom(gname).rgba[:] = [0.15, 0.30, 0.80, 1.0]

# park every free-floating body far away
for i in range(m.njnt):
    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        a = m.joint(i).qposadr[0]
        d.qpos[a:a + 3] = [-9, -9, 0.2]

# table centre is (0, 0.50); drop the two objects just above the top
pa = m.joint("penguin_joint").qposadr[0]
wa = m.joint("pick_weight_joint").qposadr[0]
d.qpos[pa:pa + 3] = [-0.15, 0.55, 0.80]
d.qpos[pa + 3:pa + 7] = [1, 0, 0, 0]
d.qpos[wa:wa + 3] = [0.15, 0.55, 0.75]
d.qpos[wa + 3:wa + 7] = [1, 0, 0, 0]

# let them settle onto the table with real physics ...
mujoco.mj_forward(m, d)
for _ in range(500):
    mujoco.mj_step(m, d)
# ... then freeze the world so nothing drifts while you look around
m.opt.gravity[:] = 0
d.qvel[:] = 0

print("Penguin and weight are on the table. Orbit with the mouse, "
      "scroll to zoom, F7 = screenshot.")
mujoco.viewer.launch(m, d)
