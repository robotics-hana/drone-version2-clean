"""Compute a wrist_cam pose (in the gripper_assembly frame) that frames the jaws +
grasped object from the open side, so the gripper is clearly visible. Renders
candidates; prints the XML pos/xyaxes for the chosen one."""
import numpy as np, mujoco, imageio.v2 as imageio
import collect_demos as C

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml"); d = mujoco.MjData(m); ik = C.ArmIK(m)
asm = m.body("gripper_assembly").id
q,off = ik.solve_drop(0.19)
d.qpos[0:3]=[0,0,0.5]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:9]=q
r=m.joint('right_clamp').qposadr[0]; l=m.joint('left_clamp').qposadr[0]; d.qpos[r]=0; d.qpos[l]=0
mujoco.mj_forward(m,d)
gs = d.site_xpos[m.site('grasp_site').id].copy()
bid=m.body('target_object').id; adr=m.jnt_qposadr[m.body_jntadr[bid]]; gid=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid][0]
m.geom_type[gid]=mujoco.mjtGeom.mjGEOM_BOX; m.geom_size[gid]=[0.010,0.006,0.025]; m.geom_rgba[gid,:3]=[0.9,0.75,0.1]
d.qpos[adr:adr+3]=gs; d.qpos[adr+3:adr+7]=[1,0,0,0]
d2=m.jnt_qposadr[m.body_jntadr[m.body('distractor_object').id]]; d.qpos[d2:d2+3]=[5,5,5]
mujoco.mj_forward(m,d)
A_pos = d.xpos[asm].copy(); A_R = d.xmat[asm].reshape(3,3)
wid = m.camera('wrist_cam').id
rend = mujoco.Renderer(m,height=480,width=640)

def make(off_world, fovy, tag):
    cam_w = gs + np.array(off_world)
    zc = -(gs - cam_w); zc = zc/np.linalg.norm(zc)          # +z_cam = -viewdir
    up = np.array([0,0,1.0]); yc = up - up.dot(zc)*zc; yc/=np.linalg.norm(yc)
    xc = np.cross(yc, zc); xc/=np.linalg.norm(xc)
    pos_rel = A_R.T @ (cam_w - A_pos)
    xr = A_R.T @ xc; yr = A_R.T @ yc
    m.cam_pos[wid]=pos_rel
    quat=np.zeros(4); Rc=np.column_stack([xc,yc,zc]); mujoco.mju_mat2Quat(quat, Rc.flatten())
    # set camera orientation via quat in parent frame
    qrel=np.zeros(4); Aq=np.zeros(4); mujoco.mju_mat2Quat(Aq, d.xmat[asm]); Aqi=np.array([Aq[0],-Aq[1],-Aq[2],-Aq[3]])
    mujoco.mju_mulQuat(qrel, Aqi, quat); m.cam_quat[wid]=qrel
    m.cam_fovy[wid]=fovy
    mujoco.mj_forward(m,d)
    rend.update_scene(d,camera='wrist_cam'); imageio.imwrite(f'wc_{tag}.png', rend.render())
    print(f'{tag}: pos_rel=[{pos_rel[0]:.4f} {pos_rel[1]:.4f} {pos_rel[2]:.4f}] '
          f'xyaxes="{xr[0]:.3f} {xr[1]:.3f} {xr[2]:.3f}  {yr[0]:.3f} {yr[1]:.3f} {yr[2]:.3f}" fovy={fovy}')

make([0,-0.13, 0.05], 65, 'front')      # open side, slightly above
make([0,-0.11, 0.10], 70, 'above')      # more above, looking down into jaws
make([0.06,-0.11,0.06], 68, 'corner')   # 3/4 view
