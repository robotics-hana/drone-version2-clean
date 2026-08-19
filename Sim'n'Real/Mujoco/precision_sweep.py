"""How precisely can the MPPI hold station? That is what gates the grasp.

Stability was the wrong metric. The controller now stays upright for 30 s, but it
wanders ~0.2 m while doing it, and the grasp needs far better: the pads close to
a 16.7 mm gap on a ~20 mm post, so lateral error at GRASP has to be within a few
millimetres. Phase arrival tolerances are 0.10 m (transit) and 0.05 m (precise),
and even those are not being met.

Measures steady-state RMS and worst-case error after a settling period, so a run
that drifts slowly is not flattered by its average.
"""
import io
import contextlib
import itertools
import numpy as np
import mujoco
from create_force_general import PureMPPIController, MPPIParams

MODEL = "SkyGrip_full.xml"


def measure(samples, horizon, w_pos, sigma_thrust, sigma_torque, torque_clip,
            seconds=12.0, settle=4.0):
    p = MPPIParams(num_samples=samples, horizon=horizon)
    p.w_pos = w_pos
    p.noise_sigma = np.array([sigma_thrust, sigma_torque, sigma_torque,
                              sigma_torque * 0.7, 0.1, 0.1])
    with contextlib.redirect_stdout(io.StringIO()):
        c = PureMPPIController(MODEL, p)
    c.torque_clip = torque_clip
    model, data = c.model, c.data
    a = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    target = np.array([0.0, 0.35, 0.95])
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = target
    data.qpos[3:7] = [1, 0, 0, 0]
    q = np.array([0.2667, -0.6933])
    data.qpos[7:9] = q
    mujoco.mj_forward(model, data)
    c.target_pos = target.copy()
    c.target_q = q.copy()
    c.arm_cmd = q.copy()

    ctrl_dt, last = 0.05, -np.inf
    u = np.zeros(6); u[0] = c.nominal_hover_thrust
    errs = []
    flipped = False
    for k in range(int(seconds / model.opt.timestep)):
        if data.time - last >= ctrl_dt:
            u = c.mppi_step(c.get_state())
            last = data.time
        c.apply_control(u)
        data.ctrl[a("act_joint1")] = q[0]
        data.ctrl[a("act_joint2")] = q[1]
        data.ctrl[a("act_gripper")] = 0.016   # fully open (was 0.025, off the new scale)
        mujoco.mj_step(model, data)
        if data.time > settle:
            errs.append(np.linalg.norm(data.qpos[0:3] - target))
        if data.qpos[2] < 0.3:
            flipped = True
            break
    if flipped or not errs:
        return None, None
    e = np.array(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


if __name__ == "__main__":
    print("steady-state station keeping, 12 s run, first 4 s discarded")
    print(f"{'ns':>4}{'H':>4}{'w_pos':>7}{'sig_T':>7}{'sig_t':>7}{'clip':>6}"
          f"{'RMS(m)':>9}{'max(m)':>9}")
    grid = [
        # baseline
        (25, 16, 100.0, 2.0, 0.03, 0.5),
        # more position weight
        (25, 16, 500.0, 2.0, 0.03, 0.5),
        (25, 16, 2000.0, 2.0, 0.03, 0.5),
        # quieter exploration
        (25, 16, 500.0, 0.5, 0.03, 0.5),
        (25, 16, 500.0, 0.5, 0.01, 0.5),
        # let it use the full torque authority
        (25, 16, 500.0, 0.5, 0.03, 1.0),
        (25, 16, 2000.0, 0.5, 0.03, 1.0),
        # more samples
        (60, 16, 500.0, 0.5, 0.03, 1.0),
    ]
    for ns, H, wp, st, sq, cl in grid:
        rms, mx = measure(ns, H, wp, st, sq, cl)
        if rms is None:
            print(f"{ns:>4}{H:>4}{wp:>7.0f}{st:>7.2f}{sq:>7.3f}{cl:>6.1f}"
                  f"{'FLIPPED':>9}{'':>9}")
        else:
            print(f"{ns:>4}{H:>4}{wp:>7.0f}{st:>7.2f}{sq:>7.3f}{cl:>6.1f}"
                  f"{rms:>9.4f}{mx:>9.4f}")
