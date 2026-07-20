"""Does the PD controller hold station with the arm reaching FORWARD?

Reaching forward throws the arm's mass sideways in y, standing up a roll moment.
The old IK docstrings say this crashed the drone -- but that was measured under
MPPI, before pd_flight.py added an exact gravity-moment feed-forward computed
from subtree_com. This re-measures the claim against the controller we actually
fly now. Station keeping must stay well inside the ~10 mm grasp window.
"""
import io, contextlib
import numpy as np, mujoco
from pd_flight import PDFlightController

MODEL = "SkyGrip_full.xml"

# (label, j1, j2) from reach_probe2: increasing forward reach, increasing comdy
POSES = [
    ("nominal  y=-0.056 comdy=+0.000", 0.340, -0.120),
    ("reach    y=-0.086 comdy=-0.024", 0.120, -0.020),
    ("reach    y=-0.118 comdy=-0.051", -0.120, 0.220),
    ("reach    y=-0.147 comdy=-0.077", -0.360, 0.540),
    ("reach    y=-0.177 comdy=-0.108", -0.680, 1.080),
]

def measure(j1, j2, seconds=14.0, settle=5.0):
    with contextlib.redirect_stdout(io.StringIO()):
        c = PDFlightController(MODEL)
    m, d = c.model, c.data
    act = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    target = np.array([0.0, 0.35, 0.95])
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = target; d.qpos[3:7] = [1, 0, 0, 0]
    q = np.array([j1, j2]); d.qpos[7:9] = q
    mujoco.mj_forward(m, d)
    c.target_pos = target.copy(); c.arm_cmd = q.copy()

    u = np.zeros(6); u[0] = c.nominal_hover_thrust
    last, errs, tilts = -np.inf, [], []
    for _ in range(int(seconds / m.opt.timestep)):
        if d.time - last >= c.CTRL_DT:
            u = c.mppi_step(c.get_state()); last = d.time
        c.apply_control(u)
        d.ctrl[act("act_gripper")] = 0.013
        d.ctrl[act("act_gripper_left")] = -0.013
        mujoco.mj_step(m, d)
        if d.time > settle:
            errs.append(np.linalg.norm(d.qpos[0:3] - target))
            R = d.xmat[c.base_id].reshape(3, 3)
            tilts.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        if d.qpos[2] < 0.3:
            return None, None, None, d.time
    e = np.array(errs)
    return float(np.sqrt(np.mean(e**2))), float(e.max()), float(np.max(tilts)), None

print("14 s hover, first 5 s discarded. Grasp window is ~10 mm.")
print(f"{'pose':<34}{'RMS(mm)':>9}{'max(mm)':>9}{'tilt(deg)':>11}")
for label, j1, j2 in POSES:
    rms, mx, tilt, tfail = measure(j1, j2)
    if rms is None:
        print(f"{label:<34}{'CRASHED at t=%.2f s' % tfail:>29}")
    else:
        print(f"{label:<34}{rms*1000:>9.2f}{mx*1000:>9.2f}{tilt:>11.2f}")
