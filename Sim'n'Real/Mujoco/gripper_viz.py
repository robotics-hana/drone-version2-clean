"""Statically pose the gripper closed on a block at the jaw centre and render the
wrist cam + a side view, so pad geometry and camera framing can be iterated fast
without running a whole episode. Poses via qpos + mj_forward (no dynamics)."""
import numpy as np, mujoco, imageio.v2 as imageio
import collect_demos as C

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml")
d = mujoco.MjData(m)
ik = C.ArmIK(m)

# Arm hanging in the overhead grasp pose; drone parked at a modest hover.
q_grasp, off = ik.solve_drop(0.19)
d.qpos[0:3] = [0.0, 0.0, 0.5]
d.qpos[3:7] = [1, 0, 0, 0]
d.qpos[7:9] = q_grasp
# Gripper joints closed (right_clamp=0 -> 15.7 mm gap; left mirrors).
r = m.joint("right_clamp").qposadr[0]; l = m.joint("left_clamp").qposadr[0]
d.qpos[r] = 0.0; d.qpos[l] = 0.0
mujoco.mj_forward(m, d)

# Put a 20x12 mm upright block centred on the jaw centre (grasp_site world pos).
gs = d.site_xpos[m.site("grasp_site").id].copy()
bid = m.body("target_object").id
adr = m.jnt_qposadr[m.body_jntadr[bid]]
gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid][0]
m.geom_size[gid] = [0.010, 0.006, 0.025]
m.geom_rgba[gid, :3] = [0.9, 0.75, 0.1]      # yellow, easy to see
d.qpos[adr:adr+3] = gs
d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
# park the distractor far away, out of frame
d2 = m.jnt_qposadr[m.body_jntadr[m.body("distractor_object").id]]
d.qpos[d2:d2+3] = [5, 5, 5]
mujoco.mj_forward(m, d)

rend = mujoco.Renderer(m, height=480, width=640)
rend.update_scene(d, camera="wrist_cam"); imageio.imwrite("viz_wrist.png", rend.render())
rend.update_scene(d, camera="scene_cam"); imageio.imwrite("viz_scene.png", rend.render())
cam = mujoco.MjvCamera(); cam.lookat[:] = gs; cam.distance = 0.20
cam.azimuth = 90; cam.elevation = -8
rend.update_scene(d, camera=cam); imageio.imwrite("viz_sideX.png", rend.render())
cam.azimuth = 0
rend.update_scene(d, camera=cam); imageio.imwrite("viz_sideY.png", rend.render())
print("jaw centre world:", gs.round(4))
print("wrote viz_wrist / viz_scene / viz_sideX / viz_sideY")
