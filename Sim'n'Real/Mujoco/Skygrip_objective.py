"""
SkyGripObjective -- MPPI ported from create_force_general.py to the full
drone + arm + gripper model (SkyGrip_full.xml, 7 actuators).

Same core algorithm as the original (sample -> rollout via real MuJoCo
dynamics -> softmax-weighted update -> receding horizon -> warm-start
shift). What changed for the port:

  1. Control is written straight to `ctrl[]` for all 7 actuators instead
     of hand-applying xfrc_applied for thrust/torque -- the current XML
     now has real `<motor site="thrust_point">` actuators for that, so
     the manual force injection is redundant and was fighting the model.
  2. Actuator indices are looked up BY NAME (model.actuator(name).id),
     not hardcoded position, per your own note that actuator/joint order
     has shifted before.
  3. Noise sigma and clip ranges are actuator-type-aware and pulled from
     the model's own ctrlrange (model.actuator_ctrlrange), not
     hand-picked constants -- arm/gripper actuators are position-type
     (radians / meters), thrust/torque actuators are motor-type
     (newtons / N*m), and mixing those with one noise scale (like the
     original 6-dim version did across a smaller set) would make the
     sampling badly mismatched now that there are 3 more, differently-
     scaled dims.
  4. Added an end-effector position cost (gripper_assembly world pos vs
     target_ee_pos), since the whole point of porting this is to reach
     for things, not just hover.

If your sbmpc stack has an actual `BaseObjective` interface to subclass,
paste it and I'll fold this into that shape exactly -- this version is
a complete, standalone, runnable class in the meantime (same
run_simulation() test harness pattern as the original).
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from dataclasses import dataclass, field


@dataclass
class MPPIParams:
    horizon: int = 10
    num_samples: int = 50
    lambda_: float = 1.0
    dt: float = 0.05

    w_pos: float = 100.0
    w_vel: float = 15.0
    w_att: float = 30.0
    w_omega: float = 10.0
    w_ee: float = 60.0          # end-effector reaching cost weight
    w_ctrl_thrust: float = 0.001
    w_ctrl_torque: float = 0.01
    w_ctrl_arm: float = 0.5     # arm/gripper are position targets -- small deviations matter more
    w_smooth_thrust: float = 0.01
    w_smooth_torque: float = 0.1
    w_smooth_arm: float = 0.2


class SkyGripObjective:
    """Drop-in successor to PureMPPIController, wired to SkyGrip_full.xml's real actuators."""

    def __init__(self, model_path: str, params: MPPIParams = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.params = params or MPPIParams()

        print(f"Creating {self.params.num_samples} simulation instances...")
        self.sim_data_pool = [mujoco.MjData(self.model) for _ in range(self.params.num_samples)]

        self.base_id = self.model.body("base_link").id
        self.gripper_id = self.model.body("gripper_assembly").id

        # End-effector reference. The gripper_assembly body ORIGIN is not where the
        # jaws close -- the clamp meshes carry large offsets, putting the origin 7.2 cm
        # behind the actual grasp point. That is wider than a 4 cm target object, so
        # aiming w_ee at the body origin converges with the object outside the jaws.
        # grasp_site sits at the jaw midpoint; fall back to the old (wrong) proxy only
        # if running against an older XML that predates the site.
        try:
            self.ee_site_id = self.model.site("grasp_site").id
        except KeyError:
            self.ee_site_id = None
            print("WARNING: no 'grasp_site' in the model -- falling back to the "
                  "gripper_assembly body origin, which is 7.2 cm off the true grasp "
                  "point. Reaching will systematically undershoot.")

        # --- actuator indices by NAME, not position ---
        act = lambda n: self.model.actuator(n).id
        self.idx_joint1 = act("act_joint1")
        self.idx_joint2 = act("act_joint2")
        self.idx_gripper = act("act_gripper")
        self.idx_thrust = act("thrust")
        self.idx_roll = act("roll_torque")
        self.idx_pitch = act("pitch_torque")
        self.idx_yaw = act("yaw_torque")

        self.arm_idx = np.array([self.idx_joint1, self.idx_joint2, self.idx_gripper])
        self.torque_idx = np.array([self.idx_roll, self.idx_pitch, self.idx_yaw])

        self.nu = self.model.nu  # 7
        self.ctrlrange = self.model.actuator_ctrlrange.copy()  # (nu, 2), read from the XML itself

        # Read from the model rather than hardcoded, so edits to arm/gripper masses
        # cannot silently desync hover thrust from the actual vehicle. (Was 1.185;
        # the model's base_link subtree is 1.1920 kg.) Note the drone does not carry
        # this rigidly -- the arm is a free pendulum, so commanding exactly m*g leaves
        # ~+0.12 m/s^2 residual. MPPI closes that; it is not an error to tune out here.
        self.total_mass = float(self.model.body_subtreemass[self.base_id])
        self.nominal_hover_thrust = self.total_mass * 9.81

        # per-actuator noise sigma, sized by what each actuator actually controls
        self.noise_sigma = np.zeros(self.nu)
        self.noise_sigma[self.idx_thrust] = 2.0
        self.noise_sigma[[self.idx_roll, self.idx_pitch]] = 0.03
        self.noise_sigma[self.idx_yaw] = 0.02
        self.noise_sigma[self.idx_joint1] = 0.05   # rad
        self.noise_sigma[self.idx_joint2] = 0.05   # rad
        self.noise_sigma[self.idx_gripper] = 0.005  # m (small -- gripper open/close is usually a discrete phase decision, not something to explore via MPPI noise)

        self.u_init = np.zeros((self.params.horizon, self.nu))
        self.u_init[:, self.idx_thrust] = self.nominal_hover_thrust
        self.noise = np.zeros((self.params.num_samples, self.params.horizon, self.nu))
        self.costs = np.zeros(self.params.num_samples)

        self.target_pos = np.array([0.0, 0.0, 1.5])       # drone position target
        self.target_ee_pos = None                          # set via set_ee_target(); None = no reaching cost

        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.1

        print("SkyGripObjective initialized:")
        print(f"  Mass: {self.total_mass:.3f} kg   Hover thrust: {self.nominal_hover_thrust:.2f} N")
        print(f"  Actuators (by name -> id): act_joint1={self.idx_joint1} act_joint2={self.idx_joint2} "
              f"act_gripper={self.idx_gripper} thrust={self.idx_thrust} "
              f"roll={self.idx_roll} pitch={self.idx_pitch} yaw={self.idx_yaw}")

    # ------------------------------------------------------------------ state
    def get_state(self, data=None):
        data = data or self.data
        pos = data.xpos[self.base_id].copy()
        vel = data.qvel[0:3].copy()
        omega = data.qvel[3:6].copy()

        R = data.xmat[self.base_id].reshape(3, 3)
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        ee_pos = (data.site_xpos[self.ee_site_id].copy()
                  if self.ee_site_id is not None
                  else data.xpos[self.gripper_id].copy())
        q_joint1 = data.qpos[self.model.joint("Joint_1").qposadr[0]]
        q_joint2 = data.qpos[self.model.joint("Joint_2").qposadr[0]]
        q_gripper = data.qpos[self.model.joint("right_clamp").qposadr[0]]

        return {
            "pos": pos, "vel": vel, "omega": omega,
            "euler": np.array([roll, pitch, yaw]),
            "ee_pos": ee_pos,
            "q_arm": np.array([q_joint1, q_joint2, q_gripper]),
        }

    # ------------------------------------------------------------- rollout
    def simulate_step(self, sim_data, control, dt):
        ctrl = np.clip(control, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        sim_data.ctrl[:] = ctrl
        n_steps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, sim_data)
        return self.get_state(sim_data)

    def compute_trajectory_cost(self, trajectory, controls):
        p = self.params
        cost = 0.0

        for t in range(len(trajectory)):
            state = trajectory[t]

            pos_error = state["pos"] - self.target_pos
            cost += p.w_pos * np.sum(pos_error ** 2)
            cost += p.w_vel * np.sum(state["vel"] ** 2)

            euler = state["euler"]
            cost += p.w_att * (euler[0] ** 2 + euler[1] ** 2 + 0.1 * euler[2] ** 2)
            cost += p.w_omega * np.sum(state["omega"] ** 2)

            if self.target_ee_pos is not None:
                ee_error = state["ee_pos"] - self.target_ee_pos
                cost += p.w_ee * np.sum(ee_error ** 2)

            if t < len(controls):
                u = controls[t]
                thrust_error = (u[self.idx_thrust] - self.nominal_hover_thrust) / self.nominal_hover_thrust
                cost += p.w_ctrl_thrust * thrust_error ** 2
                cost += p.w_ctrl_torque * np.sum(u[self.torque_idx] ** 2)
                cost += p.w_ctrl_arm * np.sum(u[self.arm_idx] ** 2)

                if t > 0:
                    du = u - controls[t - 1]
                    cost += p.w_smooth_thrust * du[self.idx_thrust] ** 2
                    cost += p.w_smooth_torque * np.sum(du[self.torque_idx] ** 2)
                    cost += p.w_smooth_arm * np.sum(du[self.arm_idx] ** 2)

            if abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5:
                cost += 500.0

            if t > 0:
                prev_err = np.linalg.norm(trajectory[t - 1]["pos"] - self.target_pos)
                curr_err = np.linalg.norm(pos_error)
                cost -= 20.0 * (prev_err - curr_err)

        return cost

    # ---------------------------------------------------------------- mppi
    def mppi_step(self, current_state):
        pos_error = np.linalg.norm(current_state["pos"] - self.target_pos)
        if pos_error > 1.0:
            self.u_init[:, self.idx_thrust] = self.nominal_hover_thrust * 1.1
            self.u_init[:, self.torque_idx] = 0
            self.u_init[:, self.arm_idx] = 0

        self.noise = np.random.randn(self.params.num_samples, self.params.horizon, self.nu) * self.noise_sigma

        saved = dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                     ctrl=self.data.ctrl.copy(), time=self.data.time)

        valid_samples = 0
        for i in range(self.params.num_samples):
            sim_data = self.sim_data_pool[i]
            sim_data.qpos[:] = saved["qpos"]
            sim_data.qvel[:] = saved["qvel"]
            sim_data.ctrl[:] = saved["ctrl"]
            sim_data.time = saved["time"]
            mujoco.mj_forward(self.model, sim_data)

            u_sample = np.clip(self.u_init + self.noise[i], self.ctrlrange[:, 0], self.ctrlrange[:, 1])

            trajectory = [current_state]
            try:
                for t in range(self.params.horizon):
                    next_state = self.simulate_step(sim_data, u_sample[t], self.params.dt)
                    trajectory.append(next_state)
                    if not np.all(np.isfinite(next_state["pos"])):
                        self.costs[i] = 1e6
                        break
                else:
                    self.costs[i] = self.compute_trajectory_cost(trajectory, u_sample)
                    valid_samples += 1
            except mujoco.FatalError as e:
                self.costs[i] = 1e6
                print(f"  sample {i}: mujoco.FatalError during rollout: {e}")

        if valid_samples == 0:
            print("WARNING: No valid samples in MPPI")
            fallback = self.u_init[0].copy()
            fallback[self.idx_thrust] = self.nominal_hover_thrust
            return fallback

        valid_mask = self.costs < 1e5
        valid_costs = self.costs[valid_mask]
        valid_noise = self.noise[valid_mask]

        min_cost = np.min(valid_costs)
        weights = np.exp(-(valid_costs - min_cost) / self.params.lambda_)
        weights /= np.sum(weights)

        weighted_noise = np.sum(weights[:, None, None] * valid_noise, axis=0)
        self.u_init = np.clip(self.u_init + weighted_noise, self.ctrlrange[:, 0], self.ctrlrange[:, 1])

        u_optimal = self.u_init[0].copy()

        self.u_init[:-1] = self.u_init[1:]
        self.u_init[-1, self.idx_thrust] = self.nominal_hover_thrust
        self.u_init[-1, self.torque_idx] = 0
        self.u_init[-1, self.arm_idx] = 0

        return u_optimal

    def apply_control(self, control):
        self.data.ctrl[:] = np.clip(control, self.ctrlrange[:, 0], self.ctrlrange[:, 1])

    # ------------------------------------------------------------- targets
    def set_target(self, pos):
        self.target_pos = np.asarray(pos, dtype=np.float64)

    def set_ee_target(self, ee_pos):
        """Cartesian target for the jaw midpoint (grasp_site) -- None disables the
        reaching cost. Note this is where a grasped object sits, NOT the
        gripper_assembly body origin, which is 7.2 cm further back."""
        self.target_ee_pos = None if ee_pos is None else np.asarray(ee_pos, dtype=np.float64)

    def set_trajectory(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        if waypoints:
            self.target_pos = waypoints[0]

    def update_target(self, current_pos):
        if not self.waypoints:
            return
        if np.linalg.norm(current_pos - self.target_pos) < self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.waypoints):
                self.target_pos = self.waypoints[self.current_waypoint_idx]

    # ---------------------------------------------------------------- run
    def run_simulation(self, duration=20.0, visualize=True):
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 4.0, -20, 45

        control_dt = 1.0 / 20
        last_control_time = 0
        self.data.qpos[2] = 1.5
        mujoco.mj_forward(self.model, self.data)
        current_control = np.zeros(self.nu)
        current_control[self.idx_thrust] = self.nominal_hover_thrust

        sim_start, real_start = self.data.time, time.time()
        mppi_time, mppi_n = 0.0, 0
        errors = []
        last_print = 0

        try:
            while self.data.time - sim_start < duration:
                t_now = self.data.time
                if t_now - last_control_time >= control_dt:
                    state = self.get_state()
                    self.update_target(state["pos"])
                    t0 = time.time()
                    current_control = self.mppi_step(state)
                    mppi_time += time.time() - t0
                    mppi_n += 1
                    last_control_time = t_now
                    errors.append(np.linalg.norm(state["pos"] - self.target_pos))

                self.apply_control(current_control)
                mujoco.mj_step(self.model, self.data)

                if t_now - last_print >= 0.5:
                    s = self.get_state()
                    print(f"t={t_now:5.1f}s pos={s['pos']} ee={s['ee_pos']} "
                          f"err={np.linalg.norm(s['pos']-self.target_pos):.3f}m "
                          f"thrust={current_control[self.idx_thrust]:.1f}N")
                    last_print = t_now
                if visualize:
                    viewer.sync()
        except KeyboardInterrupt:
            print("stopped by user")
        finally:
            if visualize:
                viewer.close()

        if mppi_n:
            print(f"avg MPPI step: {mppi_time/mppi_n*1000:.1f} ms  "
                  f"real-time factor: {(self.data.time-sim_start)/(time.time()-real_start):.2f}x")
        if errors:
            print(f"avg pos error: {np.mean(errors):.4f} m  final: {errors[-1]:.4f} m")


if __name__ == "__main__":
    controller = SkyGripObjective("SkyGrip_full.xml", MPPIParams())
    controller.set_trajectory([
        np.array([0.0, 0.0, 1.5]),
        np.array([0.3, 0.0, 1.3]),
        np.array([0.0, 0.0, 1.5]),
    ])
    controller.run_simulation(duration=20.0, visualize=True)