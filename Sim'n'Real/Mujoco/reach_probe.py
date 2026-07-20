"""What forward-reach poses exist, and do the jaws stay vertical?

Both arm joints rotate about x, so the jaws sweep the y-z plane and the closing
axis is invariant. The mouth direction is set by j1+j2. A forward reach is only
usable if there are poses that put the jaws WELL FORWARD in y while still
pointing the mouth down enough to descend onto an upright block.
"""
import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml")
d = mujoco.MjData(m)
j1a = m.joint("Joint_1").qposadr[0]; j2a = m.joint("Joint_2").qposadr[0]
sid = m.site("grasp_site").id
b = m.body("base_link").id
lo1, hi1 = m.jnt_range[m.joint("Joint_1").id]
lo2, hi2 = m.jnt_range[m.joint("Joint_2").id]
print(f"Joint_1 range {lo1:+.3f} {hi1:+.3f}   Joint_2 range {lo2:+.3f} {hi2:+.3f}")

def pose(a, c):
    d.qpos[:] = 0.0; d.qpos[3] = 1.0
    d.qpos[j1a] = a; d.qpos[j2a] = c
    mujoco.mj_forward(m, d)
    off = d.site_xpos[sid] - d.xpos[b]
    R = d.site_xmat[sid].reshape(3, 3)
    com_dy = float(d.subtree_com[m.body("Link_1").id][1] - d.xpos[b][1])
    return off, R, com_dy

# Which local axis of grasp_site points along -z (down) at the nominal grasp pose?
off0, R0, _ = pose(0.32, 0.05)
print("\ngrasp_site rotation at nominal grasp pose q=[0.32,0.05]:")
for i, nm in enumerate("xyz"):
    print(f"  local {nm} -> world {R0[:, i].round(3)}")
print(f"  jaw offset from body: {off0.round(4)}")

N = 121
g1 = np.linspace(lo1, hi1, N); g2 = np.linspace(lo2, hi2, N)
rows = []
for a in g1:
    for c in g2:
        off, R, cdy = pose(a, c)
        rows.append((a, c, off[1], off[2], R[2, 2], cdy))   # R[2,2]: how much local z is world z
rows = np.array(rows)

print("\nfull envelope: y in [%.3f, %.3f], z in [%.3f, %.3f]"
      % (rows[:, 2].min(), rows[:, 2].max(), rows[:, 3].min(), rows[:, 3].max()))

# Usable drop band, then how far forward we can get at each mouth-verticality.
drop = rows[(rows[:, 3] < -0.14) & (rows[:, 3] > -0.24)]
print(f"\nposes with jaws 0.14-0.24 m below the body: {len(drop)}")
for thr in (0.99, 0.97, 0.94, 0.90, 0.80):
    s = drop[np.abs(drop[:, 4]) > thr]
    if len(s):
        k = int(np.argmax(s[:, 2]))       # most forward (+y)
        kn = int(np.argmin(s[:, 2]))      # most back (-y)
        print(f"  |mouth.z|>{thr:.2f} ({np.degrees(np.arccos(thr)):.0f} deg tilt): "
              f"y from {s[kn,2]:+.3f} to {s[k,2]:+.3f}  "
              f"| forward pose q=[{s[k,0]:+.3f},{s[k,1]:+.3f}] z={s[k,3]:+.3f} comdy={s[k,5]:+.4f}")
    else:
        print(f"  |mouth.z|>{thr:.2f}: none")
