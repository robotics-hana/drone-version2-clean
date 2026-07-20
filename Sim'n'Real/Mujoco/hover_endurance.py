"""Can the MPPI hover hold station for 20 s with a FROZEN arm?

Every phase that ever succeeded was <=4 s. EXTEND is 20 s, and it flipped at
every arm slew rate tried (0.05 to 0.6) -- which was read as "arm motion
destabilises the drone". But static trim analysis says the extended arm costs
only 0.082 N*m of a +-1.0 N*m authority, so the arm should be a non-issue.
The alternative explanation is that the hover itself is only marginally stable
and diverges on a ~10 s timescale, independent of the arm. This separates them.
"""
import argparse
import numpy as np
import mujoco
from create_force_general import PureMPPIController, MPPIParams

MODEL = "SkyGrip_full.xml"


def run(seconds, samples, horizon, move_arm, verbose, slew=0.05):
    c = PureMPPIController(MODEL, MPPIParams(num_samples=samples, horizon=horizon))
    model, data = c.model, c.data
    a = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    surface = 0.50
    start = np.array([0.0, 0.35, surface + 0.45])
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = start
    data.qpos[3:7] = [1, 0, 0, 0]
    q_travel = np.array([0.2667, -0.6933])
    q_grasp = np.array([0.32, 0.0533])
    data.qpos[7:9] = q_travel
    mujoco.mj_forward(model, data)

    c.target_pos = start.copy()
    c.target_q = q_travel.copy()
    c.arm_cmd = q_travel.copy()
    q_cmd = q_travel.copy()

    ctrl_dt, last = 0.05, -np.inf
    u = np.zeros(6); u[0] = c.nominal_hover_thrust
    worst = 0.0
    t_fail = None
    print(f"  {'t':>5} {'x':>7} {'y':>7} {'z':>7} {'roll':>7} {'pitch':>7} {'j1':>6}")
    nsteps = int(seconds / model.opt.timestep)
    for k in range(nsteps):
        t = data.time
        if t - last >= ctrl_dt:
            u = c.mppi_step(c.get_state())
            last = t
        c.apply_control(u)
        if move_arm:
            tgt = q_grasp
            step = slew * model.opt.timestep
            q_cmd = q_cmd + np.clip(tgt - q_cmd, -step, step)
            c.target_q = tgt
        c.arm_cmd = q_cmd
        data.ctrl[a("act_joint1")] = q_cmd[0]
        data.ctrl[a("act_joint2")] = q_cmd[1]
        data.ctrl[a("act_gripper")] = 0.025
        mujoco.mj_step(model, data)

        q = data.qpos[3:7]
        roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
        pitch = np.arcsin(np.clip(2*(q[0]*q[2]-q[3]*q[1]), -1, 1))
        worst = max(worst, abs(np.degrees(roll)))
        if t_fail is None and (abs(np.degrees(roll)) > 60 or data.qpos[2] < 0.25):
            t_fail = t
        if verbose and k % int(1.0 / model.opt.timestep) == 0:
            print(f"  {t:5.1f} {data.qpos[0]:7.3f} {data.qpos[1]:7.3f} {data.qpos[2]:7.3f} "
                  f"{np.degrees(roll):7.1f} {np.degrees(pitch):7.1f} {data.qpos[7]:6.3f}")
    drift = np.linalg.norm(data.qpos[0:3] - start)
    print(f"  => arm={'MOVING' if move_arm else 'FROZEN'} | max|roll|={worst:.1f} deg | "
          f"final drift={drift:.3f} m | {'FAILED at t=%.2f' % t_fail if t_fail else 'held station'}")
    return t_fail is None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--arm", action="store_true", help="also slew the arm")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--slew", type=float, default=0.05)
    args = p.parse_args()
    run(args.seconds, args.samples, args.horizon, args.arm, not args.quiet,
        slew=args.slew)
