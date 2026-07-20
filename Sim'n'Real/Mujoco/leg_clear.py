"""Lowest point of the airframe vs the jaws, per arm pose.

Forward-reach poses hang the jaws shallower than the overhead pose, so the legs
sit relatively lower. Clearance over the work surface decides whether a reaching
grasp is flyable.

Uses the ROTATED LOCAL AABB, not geom_rbound. rbound is the bounding-SPHERE
radius, which for a long thin leg is roughly its half-length -- it reported the
legs 15 mm below the surface in the nominal pose that demonstrably flies 8/8.
"""
import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("SkyGrip_full.xml")
d = mujoco.MjData(m)
j1a = m.joint("Joint_1").qposadr[0]; j2a = m.joint("Joint_2").qposadr[0]
sid = m.site("grasp_site").id; b = m.body("base_link").id

arm_bodies = set()
def mark(bid):
    arm_bodies.add(bid)
    for i in range(m.nbody):
        if m.body_parentid[i] == bid and i != bid:
            mark(i)
mark(m.body("Link_1").id)

def geom_low(g):
    """True world-frame lowest z of a geom, via its rotated local AABB."""
    c = m.geom_aabb[g, 0:3]; h = m.geom_aabb[g, 3:6]
    R = d.geom_xmat[g].reshape(3, 3)
    centre_w = d.geom_xpos[g] + R @ c
    return centre_w[2] - float(np.abs(R[2, :]) @ h)

def survey(a, c):
    d.qpos[:] = 0.0; d.qpos[3] = 1.0
    d.qpos[j1a] = a; d.qpos[j2a] = c
    mujoco.mj_forward(m, d)
    bz = d.xpos[b][2]
    air, arm = [], []
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        bid = m.geom_bodyid[g]
        if m.body_rootid[bid] != m.body_rootid[b]:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        (arm if bid in arm_bodies else air).append((geom_low(g) - bz, nm))
    jaw = d.site_xpos[sid]
    return min(air), min(arm), jaw[2] - bz, jaw[1] - d.xpos[b][1]

POSES = [("nominal  -0.056", 0.340, -0.120), ("reach    -0.086", 0.120, -0.020),
         ("reach    -0.118", -0.120, 0.220), ("reach    -0.147", -0.360, 0.540)]

print(f"{'pose':<18}{'lowest airframe':>26}{'jaws':>9}{'leg-jaw':>9}")
for label, j1, j2 in POSES:
    (la, lan), (lm, lmn), jz, jy = survey(j1, j2)
    print(f"{label:<18}{lan:>16}{la:>10.4f}{jz:>9.4f}{la-jz:>9.4f}")

print("\nclearance over surface = (leg-jaw) + object_height - 0.010 grip depth")
print(f"{'half-height':<13}" + "".join(f"{l.split()[0]+l.split()[1]:>18}" for l, _, _ in POSES))
for hh in (0.022, 0.026, 0.030):
    row = f"{hh:<13.3f}"
    for label, j1, j2 in POSES:
        (la, _), _, jz, _ = survey(j1, j2)
        row += f"{((la - jz) + 2*hh - 0.010)*1000:>+17.1f} "
    print(row)
