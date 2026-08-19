"""Which object sizes and yaws can this gripper actually hold?

Domain randomisation is only useful if every sampled object is still graspable --
randomising into configurations the hardware cannot pick up would teach the
policy from failed demonstrations. A square post of side s presented at yaw t
spans s*(|cos t| + |sin t|), so a 20 mm post at 45 deg presents 28.3 mm. On the
Pololu gripper the jaws open to 32 mm, so that fits with only ~1.8 mm a side --
which is the limit this maps out, so randomise_episode can sample inside it.

NOTE ON THE COMMAND CONVENTION, which this script previously had backwards: the
gripper is SHUT at ctrl 0.000 and fully OPEN at 0.016 (gap_mm = 2000 * ctrl).
It used to approach at 0.025 and "close" at 0.037 -- both above the current
ctrlrange, so both now clamp to fully open and nothing would ever be gripped.

CAVEAT -- THIS HARNESS UNDER-REPORTS, and it did so before the gripper swap too.
It pins the drone's free joint every step and zeroes qvel, which is what keeps the
airframe still, but it also means the jaws TELEPORT rather than move: the contact
solver sees no relative velocity, so friction never gets a chance to carry the
object and a perfectly good grasp reads as "object never moved". Checked against
the previous gripper at HEAD~ -- it reports exactly the same nothing-lifts result
there, while collect_demos.py picks and places at 100% on this model. So treat a
negative here as "harness limitation", not "the gripper cannot grip", and trust
collect_demos.py for whether a grasp actually works.
"""
import numpy as np
import mujoco

MODEL = "SkyGrip_full.xml"
CLOSED_GAP = 0.0      # the paddles meet; see the note above


def grips(half_w, yaw, half_h=0.030, drop=0.015):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    qa = m.jnt_qposadr[m.body_jntadr[ob]]
    gid = m.body_geomadr[ob]

    m.geom_size[gid] = [half_w, half_w, half_h]
    m.body_pos[ob] = [0.0, 0.35, 0.50 + half_h]
    mujoco.mj_resetData(m, d)
    top = 0.50 + 2 * half_h
    d.qpos[qa + 3:qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    d.ctrl[aid("act_joint1")] = 0.0
    d.ctrl[aid("act_joint2")] = 0.0
    d.ctrl[aid("act_gripper")] = 0.016          # approach fully OPEN
    d.qpos[9], d.qpos[10] = 0.016, -0.016
    d.qpos[0:3] = [0, 0.35, 1.0]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    d.qpos[0:3] += (np.array([0.0, 0.35, top - drop]) - d.site_xpos[sid])
    mujoco.mj_forward(m, d)
    pin = d.qpos[0:7].copy()

    def step(n):
        for _ in range(n):
            d.qpos[0:7] = pin
            d.qvel[0:6] = 0
            mujoco.mj_step(m, d)

    step(150)
    z0 = d.qpos[qa + 2]
    if z0 < 0.45:
        return "X", 0.0                      # knocked before closing
    d.ctrl[aid("act_gripper")] = 0.000          # and SHUT to grip
    step(700)

    # Lift by RETRACTING THE ARM, not by translating the pinned body. Teleporting
    # the body while zeroing its velocity every step gives the jaws no physical
    # motion for friction to act against, so a perfectly good grasp reads as
    # "object never moved" -- that artifact is why an earlier version of this
    # test reported no grip at any size or yaw.
    # Joint_2 is the sign that raises the jaws: q_travel has it at -0.693 and
    # q_grasp at +0.053, and travel is the higher pose (TRAVEL_DROP 0.13 vs
    # GRASP_DROP 0.19), so driving Joint_2 negative retracts upward.
    for k in range(1500):
        tgt = max(-0.60, 0.0 - 0.0006 * k)
        d.ctrl[aid("act_joint2")] = tgt
        d.qpos[0:7] = pin
        d.qvel[0:6] = 0
        mujoco.mj_step(m, d)
    d.qpos[0:7] = pin
    mujoco.mj_forward(m, d)
    rise = d.qpos[qa + 2] - z0
    return ("#" if rise > 0.06 else "+" if rise > 0.02 else "." if abs(rise) < 0.01 else "x"), rise


if __name__ == "__main__":
    sides = [0.016, 0.018, 0.020, 0.022, 0.024]
    yaws = [0, 10, 20, 30, 45]
    print("held after a 0.14 m lift:  # full, + partial, . untouched, x knocked")
    print("presented width = side*(|cos|+|sin|); jaws open to 32 mm, shut to 0")
    print()
    print("side(mm) " + " ".join(f"{y:>6}d" for y in yaws))
    for s in sides:
        row = []
        for y in yaws:
            tag, _ = grips(s / 2, np.radians(y))
            pres = s * (abs(np.cos(np.radians(y))) + abs(np.sin(np.radians(y))))
            row.append(f"{tag}({pres*1000:4.1f})")
        print(f"{s*1000:7.0f}  " + " ".join(row))
