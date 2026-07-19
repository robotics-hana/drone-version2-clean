"""
SkyGripObjective(BaseObjective) -- the real sbmpc-framework version.

This replaces the earlier standalone numpy SkyGripObjective (the one that
doesn't touch sbmpc at all). This one implements the actual interface
sbmpc.solvers.RolloutGenerator expects: running_cost(state, inputs,
reference, iteration), called once per horizon step inside a
jax.lax.fori_loop, itself inside jax.vmap (over samples) inside jax.jit.
Everything here has to be trace-safe: no numpy, no python `if` on traced
values, no string-keyed lookups at call time.

state layout (mjx, non-kinematic ModelMjx): flat [qpos(nq); qvel(nv)].
For SkyGrip_full.xml: nq=18, nv=16 --
  qpos[0:3]   base_link xyz            qvel[0:3]  base_link linvel
  qpos[3:7]   base_link quat (wxyz)    qvel[3:6]  base_link angvel
  qpos[7]     Joint_1                  qvel[6]    Joint_1 dot
  qpos[8]     Joint_2                  qvel[7]    Joint_2 dot
  qpos[9]     right_clamp              qvel[8]    right_clamp dot
  qpos[10]    left_clamp               qvel[9]    left_clamp dot
  qpos[11:14] target_object xyz        qvel[10:13] target_object linvel
  qpos[14:18] target_object quat       qvel[13:16] target_object angvel
All of these addresses are looked up BY NAME at __init__ time (not
hardcoded here) and closed over as plain Python ints -- that's fine
inside jit, it's static.

Reference layout: 6-dim [drone_target_xyz(3), ee_target_xyz(3)].
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from dataclasses import dataclass, field

from sbmpc.solvers import BaseObjective


@dataclass
class SkyGripWeights:
    w_pos: float = 100.0
    w_vel: float = 15.0
    w_att: float = 30.0
    w_omega: float = 10.0
    w_ee: float = 60.0
    w_ctrl_thrust: float = 0.001
    w_ctrl_torque: float = 0.01
    w_ctrl_arm: float = 0.5
    w_final_pos: float = 300.0
    w_final_ee: float = 200.0
    tilt_limit: float = 0.5  # rad, used as a hard constraint (barrier), not a soft cost


class SkyGripObjective(BaseObjective):
    def __init__(self, mj_model_path: str, weights: SkyGripWeights = None, total_mass: float = None):
        super().__init__(robot_model=None)
        self.weights = weights or SkyGripWeights()

        # separate, self-contained mj_model/mjx_model just for index lookups
        # and the FK helper -- decoupled from whatever ModelMjx instance
        # sbmpc builds internally, so this class doesn't depend on
        # construction order.
        mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
        self.mjx_model = mjx.put_model(mj_model)
        self._fk_template = mjx.put_data(mj_model, mujoco.MjData(mj_model))

        self.base_id = mj_model.body("base_link").id
        self.gripper_id = mj_model.body("gripper_assembly").id

        # See Skygrip_objective.py: the gripper_assembly body origin sits 7.2 cm
        # behind where the jaws actually close, so it is not a usable EE reference.
        # grasp_site marks the jaw midpoint.
        try:
            self.ee_site_id = mj_model.site("grasp_site").id
        except KeyError:
            self.ee_site_id = None
            print("WARNING: no 'grasp_site' in the model -- falling back to the "
                  "gripper_assembly body origin (7.2 cm off the true grasp point).")

        self.nq = mj_model.nq

        # qpos/qvel addresses, looked up by name once (not hardcoded)
        self.i_j1 = mj_model.joint("Joint_1").qposadr[0]
        self.i_j2 = mj_model.joint("Joint_2").qposadr[0]
        self.i_rc = mj_model.joint("right_clamp").qposadr[0]
        self.i_lc = mj_model.joint("left_clamp").qposadr[0]

        # actuator indices by name
        act = lambda n: mj_model.actuator(n).id
        self.idx_thrust = act("thrust")
        self.idx_torque = jnp.array([act("roll_torque"), act("pitch_torque"), act("yaw_torque")])
        self.idx_arm = jnp.array([act("act_joint1"), act("act_joint2"), act("act_gripper")])

        if total_mass is not None:
            self.total_mass = total_mass
        else:
            # sum only the actual vehicle bodies -- NOT mocap markers
            # (current_target/waypoint_0/1 have no explicit <inertial>, so
            # mujoco defaults them to solid-sphere mass, which silently
            # added ~0.59 kg of phantom mass when I first summed
            # mj_model.body_mass blindly -- caught this in testing) and
            # NOT target_object (that's the payload being picked up, not
            # part of the vehicle's own hover-thrust budget).
            vehicle_bodies = ["base_link", "Link_1", "connect", "gripper_assembly", "clamp_1", "clamp_2"]
            self.total_mass = float(sum(mj_model.body_mass[mj_model.body(n).id] for n in vehicle_bodies))
        self.nominal_hover_thrust = self.total_mass * 9.81

    # ------------------------------------------------------------ helpers
    def _drone_state(self, state):
        pos = state[0:3]
        quat = state[3:7]  # wxyz
        vel = state[self.nq:self.nq + 3]
        omega = state[self.nq + 3:self.nq + 6]

        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # standard wxyz -> euler (matches mujoco's xmat-based roll/pitch/yaw
        # convention used elsewhere in this codebase)
        roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos, vel, jnp.array([roll, pitch, yaw]), omega

    def _ee_pos(self, state):
        """Forward kinematics through the arm chain -- needed because the
        gripper's world position is a nonlinear function of Joint_1/Joint_2,
        not something you can slice directly out of qpos.

        Returns the jaw midpoint (grasp_site), not the gripper_assembly body
        origin -- those differ by 7.2 cm, more than the width of a 4 cm target."""
        data = self._fk_template.replace(qpos=state[:self.nq])
        data = mjx.kinematics(self.mjx_model, data)
        if self.ee_site_id is not None:
            return data.site_xpos[self.ee_site_id]
        return data.xpos[self.gripper_id]

    # ---------------------------------------------------------------- cost
    def running_cost(self, state, inputs, reference, iteration):
        w = self.weights
        pos, vel, euler, omega = self._drone_state(state)
        ee_pos = self._ee_pos(state)

        target_pos = reference[0:3]
        target_ee = reference[3:6]

        cost = 0.0
        cost += w.w_pos * jnp.sum((pos - target_pos) ** 2)
        cost += w.w_vel * jnp.sum(vel ** 2)
        cost += w.w_att * (euler[0] ** 2 + euler[1] ** 2 + 0.1 * euler[2] ** 2)
        cost += w.w_omega * jnp.sum(omega ** 2)
        cost += w.w_ee * jnp.sum((ee_pos - target_ee) ** 2)

        thrust_err = (inputs[self.idx_thrust] - self.nominal_hover_thrust) / self.nominal_hover_thrust
        cost += w.w_ctrl_thrust * thrust_err ** 2
        cost += w.w_ctrl_torque * jnp.sum(inputs[self.idx_torque] ** 2)
        cost += w.w_ctrl_arm * jnp.sum(inputs[self.idx_arm] ** 2)

        # NOTE: no control-smoothness (du) term and no progress-shaping term
        # here, unlike the original numpy create_force_general.py -- this
        # running_cost signature only sees the current (state, inputs), not
        # the previous control or previous state, so neither is available
        # without a bigger change (e.g. augmenting the state with u_{t-1}).
        # Flagging this as a deliberate simplification, not an oversight.

        return cost

    def final_cost(self, state, reference, iteration):
        w = self.weights
        pos, _, _, _ = self._drone_state(state)
        ee_pos = self._ee_pos(state)
        target_pos = reference[0:3]
        target_ee = reference[3:6]
        return (w.w_final_pos * jnp.sum((pos - target_pos) ** 2)
                + w.w_final_ee * jnp.sum((ee_pos - target_ee) ** 2))

    def constraints(self, state, inputs, reference):
        _, _, euler, _ = self._drone_state(state)
        # constraint > 0 => violated => 1e3 barrier cost added (see BaseObjective.make_barrier)
        return jnp.array([
            jnp.abs(euler[0]) - self.weights.tilt_limit,
            jnp.abs(euler[1]) - self.weights.tilt_limit,
        ])