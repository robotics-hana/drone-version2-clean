"""The -y branch: how far FORWARD (toward the camera view direction) can the
jaws reach while staying vertical, and what does it cost in lateral CoM?"""
import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml")
d = mujoco.MjData(m)
j1a = m.joint("Joint_1").qposadr[0]; j2a = m.joint("Joint_2").qposadr[0]
sid = m.site("grasp_site").id; b = m.body("base_link").id

def pose(a, c):
    d.qpos[:] = 0.0; d.qpos[3] = 1.0
    d.qpos[j1a] = a; d.qpos[j2a] = c
    mujoco.mj_forward(m, d)
    off = d.site_xpos[sid] - d.xpos[b]
    R = d.site_xmat[sid].reshape(3, 3)
    return off, R[2, 2], float(d.subtree_com[m.body("Link_1").id][1] - d.xpos[b][1])

N = 161
g = np.linspace(-1.6, 1.6, N)
rows = []
for a in g:
    for c in g:
        off, mz, cdy = pose(a, c)
        rows.append((a, c, off[1], off[2], mz, cdy))
rows = np.array(rows)

# nominal, for reference
off0, mz0, cdy0 = pose(0.32, 0.05)
print(f"NOMINAL q=[0.32,0.05]: y={off0[1]:+.4f} z={off0[2]:+.4f} "
      f"tilt={np.degrees(np.arccos(abs(mz0))):.1f}deg comdy={cdy0:+.4f}")

print("\nfor each requested forward reach, the best pose at drop ~0.19 m:")
print(f"{'y_target':>9}{'j1':>8}{'j2':>8}{'y_act':>8}{'z_act':>8}{'tilt':>7}{'comdy':>8}")
for ytar in (-0.054, -0.09, -0.12, -0.15, -0.18, -0.20):
    # candidates near the target reach and near the nominal drop
    s = rows[(np.abs(rows[:, 2] - ytar) < 0.004) & (np.abs(rows[:, 3] + 0.19) < 0.012)]
    if not len(s):
        print(f"{ytar:>9.3f}   -- no pose at this reach and drop --"); continue
    k = int(np.argmin(np.abs(s[:, 5])))     # least lateral CoM among them
    a, c, y, z, mz, cdy = s[k]
    print(f"{ytar:>9.3f}{a:>8.3f}{c:>8.3f}{y:>8.4f}{z:>8.4f}"
          f"{np.degrees(np.arccos(abs(mz))):>6.1f}d{cdy:>8.4f}")

print("\nsame, but asking how deep each reach can go (max drop within 10 deg of vertical):")
for ytar in (-0.09, -0.15, -0.20):
    s = rows[(np.abs(rows[:, 2] - ytar) < 0.004) & (np.abs(rows[:, 4]) > 0.985)]
    if len(s):
        k = int(np.argmin(s[:, 3]))
        print(f"  reach {ytar:+.3f}: deepest z={s[k,3]:+.4f} at q=[{s[k,0]:+.3f},{s[k,1]:+.3f}] comdy={s[k,5]:+.4f}")
