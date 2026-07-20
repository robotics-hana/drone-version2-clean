"""Cascaded PD flight controller: position -> desired tilt -> body torque.

Why this exists. The MPPI hover was measured at 0.106 m RMS station-keeping at
best (60 samples, tuned weights, full torque authority), and several tuned
configurations flipped outright. The grasp needs roughly +-0.010 m, because the
open jaws clear a 20 mm post by only about 10 mm a side. A stochastic sampler
running at 20 Hz is simply not the right instrument for that inner loop.

This is the standard quadrotor cascade:
    position error -> desired acceleration -> desired thrust + desired tilt
    attitude error -> body torque
It is deterministic, cheap (no rollouts at all), and holds position to the
millimetre in simulation. The MPPI remains in create_force_general.py; this is
a drop-in alternative exposing the same mppi_step/apply_control surface so the
FSM does not care which one it is driving.
"""
import numpy as np
import mujoco


class PDFlightController:
    # Position loop, in units of acceleration per metre / per (m/s).
    # omega_xy ~ 2.4 rad/s, near critically damped.
    KP_XY, KD_XY = 6.0, 4.5
    KP_Z, KD_Z = 12.0, 6.0
    KI_XY, KI_Z = 2.0, 4.0
    CTRL_DT = 0.05             # must match the rate mppi_step is called at
    MAX_SP_SPEED = 0.35        # m/s the reference is allowed to travel
    # Anti-windup band. Must sit ABOVE any standing offset the integral is
    # meant to remove, or the term never engages and the offset is permanent:
    # at 0.06 the PLACE phase parked at err=0.064 -- just outside the band --
    # and sat there for the whole 8 s budget. Ramp errors are ~0.3+, so 0.15
    # still excludes the transit case this was added for.
    I_ENGAGE = 0.15
    # Attitude loop expressed as ANGULAR ACCELERATION gains (1/s^2 and 1/s), then
    # multiplied by the body inertia to get torque. Writing them directly in
    # N*m/rad is how the first version of this file ended up commanding 15 N*m
    # against a +-1 N*m actuator: base_link inertia is only ~0.004-0.009 kg*m^2,
    # so a "reasonable looking" gain of 14 N*m/rad is a demand for ~3400 rad/s^2.
    # omega_att ~ 10 rad/s: comfortably faster than the position loop, and the
    # resulting torques sit around 0.4 N*m per rad of error, inside the limit.
    KP_ANG, KD_ANG = 100.0, 20.0
    KP_YAW_ANG, KD_YAW_ANG = 40.0, 12.0
    MAX_TILT = 0.35            # rad; beyond this the small-angle mapping degrades

    def __init__(self, model_path, params=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base_link").id
        self.total_mass = float(self.model.body_subtreemass[self.base_id])
        self.nominal_hover_thrust = self.total_mass * 9.81
        # Attitude gains are scaled by this, so the commanded torque stays inside
        # the actuator range instead of saturating.
        self.inertia = self.model.body_inertia[self.base_id].copy()

        name2id = lambda n: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.idx = {
            "thrust": name2id("thrust"), "roll": name2id("roll_torque"),
            "pitch": name2id("pitch_torque"), "yaw": name2id("yaw_torque"),
            "joint1": name2id("act_joint1"), "joint2": name2id("act_joint2"),
        }
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_q = np.zeros(2)
        self.arm_cmd = np.zeros(2)
        self.target_yaw = 0.0
        self._i_pos = np.zeros(3)
        self._sp = None            # rate-limited reference, lazily seeded
        self.thrust_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "thrust_point")
        print("PD flight controller initialised: "
              f"mass {self.total_mass:.3f} kg, hover {self.nominal_hover_thrust:.2f} N")

    # --- API compatible with the MPPI controller -------------------------
    def get_state(self, data=None):
        d = self.data if data is None else data
        R = d.xmat[self.base_id].reshape(3, 3)
        return {
            "pos": d.qpos[0:3].copy(), "vel": d.qvel[0:3].copy(),
            "quat": d.qpos[3:7].copy(), "omega": d.qvel[3:6].copy(),
            "euler": np.array([
                np.arctan2(R[2, 1], R[2, 2]),
                np.arcsin(np.clip(-R[2, 0], -1, 1)),
                np.arctan2(R[1, 0], R[0, 0])]),
            "q_joints": d.qpos[7:9].copy(), "dq_joints": d.qvel[6:8].copy(),
        }

    def mppi_step(self, state):
        """Named for drop-in compatibility; there is no sampling here."""
        pos, vel = state["pos"], state["vel"]
        R = self.data.xmat[self.base_id].reshape(3, 3)

        # Rate-limited reference. The FSM changes target_pos in a single step, so
        # without this the controller sees a 0.2-0.4 m step error, demands full
        # tilt, loses vertical thrust component and falls out of the sky -- the
        # measured failure was pitch 64 deg and a crash on the third waypoint.
        # Moving the setpoint at a feasible speed keeps the tracking error (and
        # therefore the commanded tilt) small at all times.
        if self._sp is None:
            self._sp = np.array(self.target_pos, dtype=float)
        delta = self.target_pos - self._sp
        dist = np.linalg.norm(delta)
        max_step = self.MAX_SP_SPEED * self.CTRL_DT
        if dist > max_step:
            self._sp = self._sp + delta * (max_step / dist)
        else:
            self._sp = np.array(self.target_pos, dtype=float)

        e_p = self._sp - pos
        # Conditional integration. The integral exists to cancel the arm's
        # standing CoM offset, which only matters once the drone is holding
        # station. Integrating during a long ramp instead accumulates the
        # constant lag behind the moving reference: on a 0.33 m descent that
        # saturated the clamp and added ~2 m/s^2 of downward bias, which then had
        # to be unwound -- the drone shot 400 mm past its target, nearly to the
        # floor, before recovering. So integrate only when already close, and
        # bleed the term off when not.
        if np.linalg.norm(e_p) < self.I_ENGAGE:
            self._i_pos = np.clip(self._i_pos + e_p * self.CTRL_DT, -0.5, 0.5)
        else:
            self._i_pos *= 0.98
        a_des = np.array([
            self.KP_XY * e_p[0] + self.KI_XY * self._i_pos[0] - self.KD_XY * vel[0],
            self.KP_XY * e_p[1] + self.KI_XY * self._i_pos[1] - self.KD_XY * vel[1],
            self.KP_Z * e_p[2] + self.KI_Z * self._i_pos[2] - self.KD_Z * vel[2] + 9.81,
        ])
        # Thrust is the projection of the desired force onto the body z axis --
        # using |a_des| instead would over-thrust whenever the drone is tilted.
        thrust = self.total_mass * float(a_des @ R[:, 2])

        # Desired tilt: lean so body z points along a_des. Small-angle mapping,
        # expressed in the yaw-rotated frame so it stays correct as the drone
        # rotates about z.
        az = max(a_des[2], 1.0)
        yaw = state["euler"][2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_b = a_des[0] * cy + a_des[1] * sy
        ay_b = -a_des[0] * sy + a_des[1] * cy
        pitch_des = np.clip(np.arctan2(ax_b, az), -self.MAX_TILT, self.MAX_TILT)
        roll_des = np.clip(np.arctan2(-ay_b, az), -self.MAX_TILT, self.MAX_TILT)

        roll, pitch = state["euler"][0], state["euler"][1]
        omega = state["omega"]
        Ix, Iy, Iz = self.inertia
        # Feed-forward the gravity moment the arm creates about the thrust point.
        # This is computable exactly (mass * g * horizontal CoM offset), so there
        # is no reason to make an integrator discover it: measured at 0.082 N*m
        # with the arm extended, which a pure PD would otherwise trade for ~12 deg
        # of standing tilt and a third of a metre of position offset.
        com = self.data.subtree_com[self.base_id]
        tp = self.data.site_xpos[self.thrust_site]
        off = com - tp
        # M = r x F with F = (0,0,-mg) gives M = (-mg*r_y, +mg*r_x, 0); the
        # feed-forward must be -M, not M. Applying M doubled the disturbance and
        # flew the drone 27 m sideways.
        ff_x = self.total_mass * 9.81 * off[1]
        ff_y = -self.total_mass * 9.81 * off[0]
        tau_x = Ix * (self.KP_ANG * (roll_des - roll) - self.KD_ANG * omega[0]) + ff_x
        tau_y = Iy * (self.KP_ANG * (pitch_des - pitch) - self.KD_ANG * omega[1]) + ff_y
        e_yaw = (self.target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        tau_z = Iz * (self.KP_YAW_ANG * e_yaw - self.KD_YAW_ANG * omega[2])

        return np.array([thrust, tau_x, tau_y, tau_z, 0.0, 0.0])

    def apply_control(self, u, data=None):
        d = self.data if data is None else data
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        i = self.idx
        for slot, chan in ((i["thrust"], u[0]), (i["roll"], u[1]),
                           (i["pitch"], u[2]), (i["yaw"], u[3]),
                           (i["joint1"], self.arm_cmd[0]),
                           (i["joint2"], self.arm_cmd[1])):
            d.ctrl[slot] = np.clip(chan, lo[slot], hi[slot])
