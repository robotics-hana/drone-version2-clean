"""Does the gripper actually grip the target object?

The flight controller and the gripper are independent failure modes, and chasing
them together is what made the last few rounds slow. This pins the drone rigidly
in place (re-imposing the free-joint pose every step, so it cannot fall) and asks
one question: with the jaws straddling the object, does closing them and then
retracting the arm carry the object with it?

Sweeps the closed-gripper command, because the aperture-vs-ctrl mapping is set by
mesh geometry that is not safe to read off the XML by eye.

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


def probe(close_cmd, straddle_z, verbose=False):
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)

    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    obj_b = bid("target_object")
    obj_q = model.body_jntadr[obj_b]
    obj_qadr = model.jnt_qposadr[obj_q]

    a_j1, a_j2 = aid("act_joint1"), aid("act_joint2")
    a_gr = aid("act_gripper")
    q_j1 = model.jnt_qposadr[jid("Joint_1")]
    q_j2 = model.jnt_qposadr[jid("Joint_2")]

    mujoco.mj_resetData(model, data)
    obj0 = data.qpos[obj_qadr:obj_qadr + 3].copy()

    # Arm straight down, gripper open.
    data.ctrl[a_j1] = 0.0
    data.ctrl[a_j2] = 0.0
    data.ctrl[a_gr] = 0.016            # fully open: 32 mm gap (Pololu kit's range)
    data.qpos[q_j1] = 0.0
    data.qpos[q_j2] = 0.0
    mujoco.mj_forward(model, data)

    # Place the DRONE so that grasp_site lands at the requested height directly
    # over the object. grasp_site is measured, not assumed -- solve for the body
    # offset that puts it where we want it.
    drone_pos = np.array([obj0[0], obj0[1], 1.0])
    data.qpos[0:3] = drone_pos
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    site_now = data.site_xpos[sid].copy()
    # shift the body by the residual so the site is exactly on target
    want = np.array([obj0[0], obj0[1], straddle_z])
    drone_pos += (want - site_now)
    data.qpos[0:3] = drone_pos
    mujoco.mj_forward(model, data)

    pinned_qpos = data.qpos[0:7].copy()

    def pin():
        data.qpos[0:7] = pinned_qpos
        data.qvel[0:6] = 0.0

    def run(nsteps):
        for _ in range(nsteps):
            pin()
            mujoco.mj_step(model, data)
        pin()
        mujoco.mj_forward(model, data)

    run(300)                       # settle with jaws open
    z_before = data.qpos[obj_qadr + 2]
    site_before = data.site_xpos[sid].copy()

    data.ctrl[a_gr] = close_cmd    # close
    run(500)
    z_closed = data.qpos[obj_qadr + 2]

    # Lift by raising the PINNED DRONE, not by retracting the arm. Arm retraction
    # depends on a joint sign convention that is easy to get backwards (the first
    # version of this probe drove the jaws 33 mm DOWNWARD); moving the pin is
    # unambiguous -- up is up.
    for _ in range(600):
        pinned_qpos[2] += 0.00025      # 0.15 m over the lift
        pin()
        mujoco.mj_step(model, data)
    pin()
    mujoco.mj_forward(model, data)
    z_after = data.qpos[obj_qadr + 2]
    site_after = data.site_xpos[sid].copy()

    lifted = z_after - z_before
    jaw_rise = site_after[2] - site_before[2]
    if verbose:
        print(f"    site z {site_before[2]:.3f} -> {site_after[2]:.3f} "
              f"(rise {jaw_rise:+.3f}) | object z {z_before:.3f} -> {z_after:.3f}")
    return lifted, jaw_rise, z_closed - z_before


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    qa = model.jnt_qposadr[model.body_jntadr[b]]
    obj_z = data.qpos[qa + 2]
    gid = model.body_geomadr[b]
    half = model.geom_size[gid][2]
    print(f"object centre z={obj_z:.3f} half-height={half:.3f} "
          f"spans {obj_z-half:.3f}..{obj_z+half:.3f}")
    print(f"gripper ctrlrange = {model.actuator_ctrlrange[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,'act_gripper')]}")
    print()

    # Straddle heights: jaws centred on the post, and progressively lower.
    for straddle in (obj_z + half, obj_z + half / 2, obj_z, obj_z - half / 2):
        print(f"grasp_site z = {straddle:.3f} "
              f"({straddle-obj_z:+.3f} vs object centre)")
        for cmd in (0.000, 0.005, 0.008, 0.010, 0.014):
            lifted, jaw_rise, squeeze = probe(cmd, straddle)
            tag = "LIFTED" if lifted > 0.02 else ("moved" if abs(lifted) > 0.002 else "-")
            print(f"    close={cmd:.3f} | object moved {lifted:+.4f} m "
                  f"(jaws rose {jaw_rise:+.3f}) {tag}")
        print()


if __name__ == "__main__":
    main()
