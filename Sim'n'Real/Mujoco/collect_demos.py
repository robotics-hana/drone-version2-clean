"""
SkyGrip demo collection -> LeRobotDataset (v3.0 API)

Scripts a pick-and-place trajectory per episode, randomises object pose,
drone start pose and lighting, and records wrist + scene frames, state and
action to a LeRobotDataset for SmolVLA finetuning.

Run:
    python collect_demos.py --episodes 50 --out_repo_id yourname/skygrip_pick_place


ACTION SPACE (6-D)
------------------
    [drone_x, drone_y, drone_z, joint1, joint2, gripper]

The VLA emits SETPOINTS at chunk rate; MPPI realises them at control rate.
It does NOT emit thrust/torque. Three reasons:

  1. Thrust and body torques are only meaningful relative to instantaneous
     attitude and mass. A chunk of them replayed even slightly out of phase
     is not just degraded, it is unstable -- exactly the discontinuity
     problem RTC exists to paper over.
  2. Setpoints are what the platform's own controller consumes. AirVLA does
     the same thing, emitting position setpoints to PX4; here MPPI plays the
     role PX4 plays there, with the advantage that it optimises over the
     coupled drone+arm dynamics rather than needing a payload-compensation
     patch bolted on at inference time.
  3. Joint-space for the manipulator is the dissertation's stated choice.
     The drone body has no joints, so its "joint space" IS its pose -- a
     Cartesian body setpoint does not contradict that decision, whereas an
     end-effector-space target for the ARM would.

The gripper channel is continuous here (metres of jaw travel) because that
is what the actuator takes, but it is driven as an effectively binary
open/closed phase decision. If the abstain head later needs a discrete
grasp token, that is a separate output, not a reinterpretation of this one.


OBSERVATION STATE (16-D)
------------------------
    [joint1, joint2, gripper, pos(3), quat(4), linvel(3), angvel(3)]

Drone pose and twist ARE included. This was flagged as an open design
question, and the answer follows from the geometry: the gripper's world
position is a function of drone pose AND arm angles, so arm angles alone are
not a Markov state for a manipulation task on a moving base. A fixed-base
SmolVLA setup gets away with joints-only precisely because its base pose is
constant. Ours is not.

Cost: smolvla_base was pretrained with state dim 6. Anything up to 32 is
accepted and zero-padded (pad_vector pads but never truncates -- 33 would
raise), but the further from 6, the less the pretrained input projection
transfers. 16 is a deliberate trade, not an oversight.


CAMERAS
-------
Keys MUST be observation.images.camera1/camera2/... . smolvla_base ships
non-empty input_features naming camera1/camera2/camera3, and factory.py only
infers feature names from the dataset `if not cfg.input_features:` -- which
is false for this checkpoint. Dataset visuals must therefore be a subset of
{camera1, camera2, camera3}. Naming them "wrist"/"scene" fails
validate_visual_features_consistency outright. The mapping to physical
cameras is recorded in CAMERAS below.


WHY THE ARM IS SERVOED, NOT SCRIPTED
------------------------------------
The single hardest constraint on this platform: the grasp window is ~5 mm tall
(measured -- the band where the jaws both clear the object on approach and can
close on it), while MPPI holds body position to ~70 mm. The body controller is
an order of magnitude coarser than the task.

Open-loop arm angles therefore cannot work, however good the IK is. The arm
servos track setpoints to ~1 mm and Joint_2 directly sets the vertical drop, so
the arm re-solves against the drone's MEASURED pose every control step and
absorbs the body's error (SkyGripController.step, `servo_to`). Measured clean
approaches: 78% with the body within +-100 mm, 89% within +-70 mm, 100% within
+-50 mm -- hence ARRIVE_TOL_PRECISE.

This is a genuine difference from fixed-base SmolVLA setups, where the base
never moves and the arm never has to reject base disturbance.

STILL UNVERIFIED
----------------
Everything above is validated STATICALLY -- poses stepped through mj_forward and
checked for contact/clearance. A full flying episode ending in a successful lift
has NOT been demonstrated. The last end-to-end attempt fell, though its cause is
understood and fixed (the old geometry drove the jaws 28 mm into the pedestal and
the contact impulse threw the drone). Run a single episode and watch it before
trusting a 50-episode batch.
"""

import argparse
import numpy as np
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from create_force_general import PureMPPIController, MPPIParams

MODEL_PATH = "SkyGrip_full.xml"
FPS = 30

# SmolVLA resizes to 512x512 internally (resize_imgs_with_padding). Logging
# smaller just means upscaling later with no detail to recover; 480x640 keeps
# real resolution without bloating the dataset.
IMG_H, IMG_W = 480, 640

# LeRobot feature key -> MuJoCo camera name. The keys are fixed by the
# checkpoint (see CAMERAS note above); the values are ours to choose.
CAMERAS = {
    "camera1": "wrist_cam",    # manipulation close-up, top-down over the jaws
    "camera2": "scene_cam",    # forward/navigation view from the nose
}

PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE"]

STATE_NAMES = (
    ["joint1", "joint2", "gripper"]
    + [f"drone_pos_{a}" for a in "xyz"]
    + [f"drone_quat_{a}" for a in "wxyz"]
    + [f"drone_linvel_{a}" for a in "xyz"]
    + [f"drone_angvel_{a}" for a in "xyz"]
)
ACTION_NAMES = ["drone_x", "drone_y", "drone_z", "joint1", "joint2", "gripper"]

# Gripper aperture, in metres of right_clamp travel. These map to the TRUE
# inner-face gap between the jaws, measured from the clamp meshes:
#     0.037 -> 30.7 mm   (fully open; the widest object the jaws can admit)
#     0.031 -> ~19 mm
#     0.025 ->   6.7 mm
#     0.012 -> jaws overlapping
# Do NOT size objects from the distance between clamp geom ORIGINS -- that reads
# 87.2 mm at full aperture and is meaningless, the same error as using the
# gripper_assembly body origin as the end effector.
# The target post is 20 mm wide, so CLOSED squeezes it by ~1 mm to hold it.
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.031

# Phase advance. Two tolerances, because the phases have different needs:
#   TRANSIT  -- matches the MPPI's own waypoint tolerance; asking for tighter than
#               the controller resolves just makes every phase time out.
#   PRECISE  -- used only where the jaws must land on something. Measured: with
#               closed-loop arm compensation, approaches are clean 78% of the time
#               if the body is held within +-100 mm, 89% within +-70 mm, and
#               100% within +-50 mm. The residual failures are the jaw taper (a
#               mesh property, not tunable), so holding tighter is the only lever.
# HOLD_FRAMES requires the drone to STAY inside tolerance rather than clip through
# it at speed, which is what makes the grasp repeatable. MAX caps a phase that
# cannot converge so one bad episode does not stall a 50-episode run.
ARRIVE_TOL_TRANSIT = 0.10
ARRIVE_TOL_PRECISE = 0.05
HOLD_FRAMES = 8
MAX_FRAMES_PER_PHASE = int(4.0 * FPS)


class SkyGripController:
    """Drives the drone with the Feedback-MPPI from create_force_general.py.

    Owns the MjModel/MjData (PureMPPIController builds them from the path), so
    the collector renders and reads state from THESE, not a second copy -- two
    MjData for one episode would silently diverge.

    Division of labour:
      body            -- MPPI (thrust + body torques)
      arm and gripper -- written straight to ctrl

    The arm deliberately does NOT go through MPPI's sampler. Its actuators are
    stiff position servos (kp=998), so a setpoint written to ctrl is tracked
    almost exactly; routed through MPPI instead, the joint channels get averaged
    against the position/attitude terms and barely move -- measured j1=0.354
    against a commanded 0.90-1.20. mppi.target_q is still set so the rollouts
    anticipate where the arm is going, but the command itself is direct.
    """

    CONTROL_HZ = 20.0        # MPPI replan rate; physics runs at 1 kHz

    def __init__(self, model_path, params=None):
        self.mppi = PureMPPIController(model_path, params or MPPIParams())
        self.model = self.mppi.model
        self.data = self.mppi.data

        self.idx_arm = [self.model.actuator(n).id
                        for n in ("act_joint1", "act_joint2")]
        self.idx_gripper = self.model.actuator("act_gripper").id
        self.drone_target = np.array([0.0, 0.0, 1.5])
        self.joint_target = np.zeros(2)
        self.gripper_cmd = GRIPPER_OPEN

        self._ctrl_dt = 1.0 / self.CONTROL_HZ
        self._last_ctrl = -np.inf
        self._u = np.zeros(6)
        self._u[0] = self.mppi.nominal_hover_thrust
        self._servo_to = None
        self._ik = None

    def set_targets(self, drone_xyz, joints, gripper, servo_to=None, ik=None):
        """servo_to: optional WORLD point the jaws should hold, closed-loop.

        Without it the arm just holds `joints` open-loop, which is fine in transit.
        With it, the arm re-solves every control step against the drone's MEASURED
        altitude -- see step(). That matters because the grasp window is ~5 mm tall
        while MPPI holds position to ~70 mm: open-loop arm angles miss by an order
        of magnitude more than the window allows, no matter how good the IK is.
        The arm servos track to ~1 mm, so letting them absorb the body's error is
        the only way the grasp closes at all.
        """
        self.drone_target = np.asarray(drone_xyz, dtype=np.float64)
        self.joint_target = np.asarray(joints, dtype=np.float64)
        self.gripper_cmd = float(gripper)
        self.mppi.target_pos = self.drone_target
        self.mppi.target_q = self.joint_target
        self._servo_to = None if servo_to is None else np.asarray(servo_to, float)
        self._ik = ik

    def reset_after_randomisation(self):
        """Re-seed the MPPI warm start after qpos is overwritten.

        u_init carries the previous episode's solution. Left alone, the first
        control steps of a new episode are optimising against a pose that no
        longer exists, which shows up as a lurch in the opening frames -- i.e.
        contaminating exactly the part of every episode the policy sees most.
        """
        self.mppi.u_init[:] = 0.0
        self.mppi.u_init[:, 0] = self.mppi.nominal_hover_thrust
        self.mppi.u_prev = np.zeros(self.mppi.nu)
        self._last_ctrl = -np.inf
        self._u = np.zeros(6)
        self._u[0] = self.mppi.nominal_hover_thrust

    def step(self):
        # Replan at CONTROL_HZ, not every physics step -- an MPPI solve is
        # ~1.8 s of wall clock, so calling it per mj_step would be 1000x the
        # necessary cost for no benefit.
        if self.data.time - self._last_ctrl >= self._ctrl_dt:
            self._u = self.mppi.mppi_step(self.mppi.get_state())
            self._last_ctrl = self.data.time

            # Closed-loop arm compensation. Re-solve the arm against where the
            # drone ACTUALLY is, not where it was asked to be, so the jaws stay on
            # the target while the body drifts inside its 0.10 m tolerance.
            if self._servo_to is not None and self._ik is not None:
                want = self._servo_to - self.data.qpos[0:3]
                want[0] = self._ik.x_offset      # arm cannot move in x; do not ask
                q, err = self._ik.solve(want)
                if err < 0.02:                   # ignore solves outside the envelope
                    self.joint_target = q
                    self.mppi.target_q = q

        self.mppi.apply_control(self._u)
        # Overwrite the arm/gripper channels AFTER apply_control: MPPI's solution
        # for them is discarded in favour of the commanded setpoint. MPPI re-plans
        # from measured state every step, so it absorbs the difference rather than
        # fighting it.
        self.data.ctrl[self.idx_arm[0]] = self.joint_target[0]
        self.data.ctrl[self.idx_arm[1]] = self.joint_target[1]
        self.data.ctrl[self.idx_gripper] = self.gripper_cmd

    def action(self):
        """Logged action = the COMMANDED setpoints, not MPPI's realised output.

        The VLA is being trained to emit setpoints for MPPI to track, so the
        supervision target is the setpoint. Logging MPPI's thrust/torque instead
        would train the policy to imitate the low-level controller.
        """
        return np.concatenate([self.drone_target, self.joint_target,
                               [self.gripper_cmd]]).astype(np.float32)


def randomise_episode(model, data, rng):
    """Domain-randomise object, drone start pose and lighting.

    Volume alone overfits: LeRobot's own guidance pairs "~50 episodes" with 5
    distinct object positions x 10 episodes, and documents 25 episodes as too
    few. Randomise per episode, not per batch of episodes.
    """
    mujoco.mj_resetData(model, data)

    # --- pedestal: static body, so it moves by writing model.body_pos rather
    #     than qpos. Shifting the whole work surface (not just the object on it)
    #     stops the policy learning one fixed approach corridor.
    ped = model.body("pedestal").id
    ped_y = rng.uniform(0.28, 0.42)
    model.body_pos[ped] = [0.0, ped_y, 0.25]
    surface_z = 0.25 + model.geom_size[
        [g for g in range(model.ngeom) if model.geom_bodyid[g] == ped][0]][2]

    # --- object: on the pedestal surface, within its x extent, plus yaw so the
    #     grasp is not always axis-aligned (a fixed yaw teaches one approach only)
    bid = model.body("target_object").id
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    # object centre sits half its height above the surface (block is 120 mm tall)
    obj_half_h = model.geom_size[
        [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]][2]
    obj = np.array([rng.uniform(-0.14, 0.14), ped_y, surface_z + obj_half_h])
    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    data.qpos[adr:adr + 3] = obj
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # --- drone start pose: vary where the episode begins, so the policy sees
    #     approach from a spread of offsets rather than one canned trajectory.
    #     Start above the surface, not at the old 1.5 m -- the grasp happens at
    #     ~0.75 m, and starting a metre high just spends every episode descending.
    data.qpos[0:3] = [rng.uniform(-0.12, 0.12),
                      ped_y + rng.uniform(-0.15, 0.15),
                      surface_z + rng.uniform(0.38, 0.52)]

    # --- lighting: diffuse level and sun direction. Cheap to vary and it is
    #     what shifts most between sim and any real deployment.
    sun = model.light("sunlight").id
    fill = model.light("ambient_fill").id
    lvl = rng.uniform(0.55, 1.00)
    model.light_diffuse[sun] = [lvl, lvl, lvl]
    model.light_dir[sun] = _unit([rng.uniform(-0.4, 0.4),
                                  rng.uniform(-0.4, 0.4),
                                  -1.0])
    amb = rng.uniform(0.15, 0.40)
    model.light_diffuse[fill] = [amb, amb, amb]

    # --- object colour, so the policy keys on shape/position not one RGB
    gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]
    model.geom_rgba[gid, :3] = rng.uniform(0.15, 0.9, size=3)

    mujoco.mj_forward(model, data)
    return obj


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def get_state(model, data):
    """16-D proprioceptive state. Joints looked up by name -- qpos layout has
    shifted before and will again if bodies are added."""
    j = lambda n: data.qpos[model.joint(n).qposadr[0]]
    return np.concatenate([
        [j("Joint_1"), j("Joint_2"), j("right_clamp")],
        data.qpos[0:3],      # drone position
        data.qpos[3:7],      # drone orientation (quat, wxyz)
        data.qvel[0:3],      # linear velocity
        data.qvel[3:6],      # angular velocity
    ]).astype(np.float32)


def grasp_site_pos(model, data):
    """True grasp point: the jaw midpoint, NOT the gripper_assembly body
    origin (those differ by 7.2 cm -- wider than the 40 mm target)."""
    return data.site_xpos[model.site("grasp_site").id].copy()


class ArmIK:
    """Numeric IK for Joint_1/Joint_2 against the grasp site.

    Solved by search over MuJoCo's own forward kinematics rather than derived
    analytically. The chain from base_link to the jaws runs through two 90-degree
    quaternions plus a 7.2 cm mesh offset; a hand-derived 2-link solution has to
    reproduce all of that exactly and silently returns wrong angles if it does
    not. FK is the ground truth the simulator will actually use, so searching it
    cannot disagree with the physics.

    Only 2 DOF, so a coarse grid plus local refinement is exact enough and needs
    no solver dependency. The map is built once against a scratch MjData with
    the drone at the origin -- the arm's offset from the body is independent of
    where the body is, so one table serves every episode.
    """

    def __init__(self, model, coarse=61, refine_steps=3):
        self._m = model
        self._d = mujoco.MjData(model)
        self._j1 = model.joint("Joint_1").qposadr[0]
        self._j2 = model.joint("Joint_2").qposadr[0]
        self._sid = model.site("grasp_site").id
        lo1, hi1 = model.jnt_range[model.joint("Joint_1").id]
        lo2, hi2 = model.jnt_range[model.joint("Joint_2").id]
        self._lim = ((lo1, hi1), (lo2, hi2))
        self._refine_steps = refine_steps

        g1 = np.linspace(lo1, hi1, coarse)
        g2 = np.linspace(lo2, hi2, coarse)
        self._grid = np.array([(a, b) for a in g1 for b in g2])
        self._offsets = np.array([self._offset(a, b) for a, b in self._grid])

        # Both joints rotate about x, so the arm sweeps the y-z plane ONLY -- it
        # cannot move the jaws laterally at all. The jaw centre sits at a fixed
        # x offset from the drone's centreline (the gripper is mounted off-centre),
        # and every IK target must respect it: asking for x=0 asks for something no
        # joint can produce, and the solver returns its closest pose with a residual
        # of exactly that offset. The DRONE has to absorb it by flying x_offset to
        # one side of the object.
        self.x_offset = float(np.mean(self._offsets[:, 0]))
        spread = float(np.ptp(self._offsets[:, 0]))
        assert spread < 1e-6, f"arm moves in x by {spread:.4f} m -- x_offset is not constant"

    def _offset(self, j1, j2):
        """grasp_site position relative to the drone body, in body coordinates."""
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0                      # identity quaternion
        d.qpos[self._j1] = j1
        d.qpos[self._j2] = j2
        mujoco.mj_kinematics(self._m, d)
        return d.site_xpos[self._sid] - d.xpos[self._m.body("base_link").id]

    def solve(self, target_offset):
        """Joint angles putting the jaws at `target_offset` from the body.

        Returns (angles, residual). CHECK THE RESIDUAL -- targets outside the
        arm's envelope return the closest reachable pose rather than failing, so
        a caller that ignores it will silently command a pose that does not
        reach the object.
        """
        t = np.asarray(target_offset, dtype=np.float64)
        k = int(np.argmin(np.linalg.norm(self._offsets - t, axis=1)))
        best = self._grid[k].copy()
        best_err = np.linalg.norm(self._offsets[k] - t)

        span = (self._lim[0][1] - self._lim[0][0]) / len(np.unique(self._grid[:, 0]))
        for _ in range(self._refine_steps):
            improved = False
            for d1 in (-span, 0.0, span):
                for d2 in (-span, 0.0, span):
                    a = np.clip(best[0] + d1, *self._lim[0])
                    b = np.clip(best[1] + d2, *self._lim[1])
                    err = np.linalg.norm(self._offset(a, b) - t)
                    if err < best_err - 1e-9:
                        best, best_err, improved = np.array([a, b]), err, True
            span *= 0.5
            if not improved:
                span *= 0.5
        return best, best_err


def build_plan(ik, obj, place, obj_half_height):
    """Per-phase (drone target, joint target, gripper) with joints from IK.

    The drone grasps from directly overhead. GRASP_DROP is the vertical distance
    from the body to the jaws, so the body flies to object_z + GRASP_DROP and the
    jaws land on the object. TRAVEL_DROP tucks the arm up between waypoints so
    the payload is not swinging at full extension during transit.
    """
    # GRASP_DROP is the NOMINAL body-to-jaw distance; the arm swings around it to
    # absorb body error. Deliberately well short of the 0.271 m envelope: the
    # headroom below it is how far the arm can reach down when the drone ends up
    # HIGH, and running out of it is an uncorrectable miss. At 0.23 there were
    # only 41 mm spare and a +80 mm body error left a 39 mm residual; 0.19 gives
    # ~81 mm, covering MPPI's error band in both directions. Costs leg clearance
    # (legs sit 0.20 m below the body) but still leaves ~100 mm over the pedestal.
    GRASP_DROP, TRAVEL_DROP = 0.19, 0.13
    CRUISE = 0.42                      # body height above the surface in transit

    # Put the jaw midpoint just ABOVE the object's top face. The gripper housing
    # obstructs from ~0 mm above the midpoint, so the object must sit entirely
    # BELOW it -- aiming at the object's centre buries its top half in the housing.
    # 5 mm of clearance above the top face; the fingers then grip the 44 mm of the
    # block immediately beneath, and the fingertips (49.2 mm down) still clear the
    # surface by ~56 mm. Derived from the object's half-height so it stays correct
    # if the block is resized.
    GRASP_RISE = obj_half_height + 0.030

    # Solve at the arm's fixed lateral offset -- x is not a free variable.
    q_grasp, e_grasp = ik.solve([ik.x_offset, 0.0, -GRASP_DROP])
    q_travel, e_travel = ik.solve([ik.x_offset, 0.0, -TRAVEL_DROP])
    for name, err in (("grasp", e_grasp), ("travel", e_travel)):
        if err > 0.01:
            raise RuntimeError(
                f"IK for the {name} pose is {err*1000:.0f} mm off -- outside the "
                f"arm's envelope. Adjust GRASP_DROP/TRAVEL_DROP.")

    # Body target = where the jaws must be, minus the body-to-jaw offset. The x
    # term is what makes the drone fly slightly to one side so the off-centre
    # gripper lands on the object rather than beside it.
    to_body = np.array([-ik.x_offset, 0.0, GRASP_RISE + GRASP_DROP])
    over_obj = obj + np.array([-ik.x_offset, 0.0, CRUISE])
    at_obj = obj + to_body
    over_place = place + np.array([-ik.x_offset, 0.0, CRUISE])
    at_place = place + to_body

    # Phases that must actually land on the object carry a world-space servo
    # point; transit phases do not need one.
    grasp_pt = obj + np.array([0.0, 0.0, GRASP_RISE])
    place_pt = place + np.array([0.0, 0.0, GRASP_RISE])
    return {
        "APPROACH":  (over_obj,   q_travel, GRIPPER_OPEN, None),
        "DESCEND":   (at_obj,     q_grasp,  GRIPPER_OPEN, grasp_pt),
        "GRASP":     (at_obj,     q_grasp,  GRIPPER_CLOSED, grasp_pt),
        "LIFT":      (over_obj,   q_travel, GRIPPER_CLOSED, None),
        "TRANSPORT": (over_place, q_travel, GRIPPER_CLOSED, None),
        "PLACE":     (at_place,   q_grasp,  GRIPPER_CLOSED, place_pt),
        "RELEASE":   (at_place,   q_grasp,  GRIPPER_OPEN, place_pt),
    }


def run_episode(model, data, renderer, controller, ik, rng, task_text):
    obj = randomise_episode(model, data, rng)
    controller.reset_after_randomisation()

    # place target: same surface, offset along the pedestal's long axis
    place = np.array([obj[0] + rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.16),
                      obj[1], obj[2]])
    place[0] = np.clip(place[0], -0.16, 0.16)
    obj_half_height = model.geom_size[
        [g for g in range(model.ngeom)
         if model.geom_bodyid[g] == model.body('target_object').id][0]][2]
    plan = build_plan(ik, obj, place, obj_half_height)

    frames = []
    # Physics runs at model timestep (1 ms); a frame is one 1/FPS interval, so
    # each recorded frame must advance the sim by 1/FPS of PHYSICS, not by a
    # single mj_step. Stepping once per frame would record 210 frames spanning
    # 0.21 s while declaring fps=30 (i.e. 7 s) -- every timestamp wrong by 33x,
    # and the arm would barely move within an episode.
    #
    # 1 ms does not divide 1/30 s evenly, so substeps rounds 33.33 -> 33 and an
    # episode spans 6.93 s of sim rather than 7.00 (a uniform 1% rate error).
    # Harmless: LeRobot derives timestamps as frame_index/fps, so the dataset is
    # internally consistent; only sim-time and declared-time differ. Set FPS=25
    # (40 substeps) or timestep=1/30000 if you ever need them to agree exactly.
    substeps = max(1, round((1.0 / FPS) / model.opt.timestep))

    for phase in PHASES:
        drone_xyz, joints, grip, servo_to = plan[phase]
        controller.set_targets(drone_xyz, joints, grip, servo_to=servo_to, ik=ik)

        # Phases advance on ARRIVAL, not on a timer. A fixed 1 s per phase cuts
        # the drone off mid-transit -- APPROACH alone can be a 1 m descent, which
        # MPPI cannot complete in a second -- so every later phase then starts
        # from the wrong place and the grasp misses for reasons that have nothing
        # to do with the IK. MIN keeps a few frames even when already on target
        # (so GRASP/RELEASE still show the jaws moving); MAX stops a phase that
        # cannot converge from hanging the whole run.
        held = 0
        for f_i in range(MAX_FRAMES_PER_PHASE):
            for _ in range(substeps):
                controller.step()
                mujoco.mj_step(model, data)

            tol = ARRIVE_TOL_PRECISE if servo_to is not None else ARRIVE_TOL_TRANSIT
            if np.linalg.norm(data.qpos[0:3] - drone_xyz) < tol:
                held += 1
            else:
                held = 0

            frame = {
                "observation.state": get_state(model, data),
                "action": controller.action(),
                "task": task_text,
            }
            for key, cam in CAMERAS.items():
                renderer.update_scene(data, camera=cam)
                frame[f"observation.images.{key}"] = renderer.render().copy()
            frames.append(frame)

            if held >= HOLD_FRAMES:
                break

    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out_repo_id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=str,
                    default="pick up the red cube and place it to the side")
    ap.add_argument("--samples", type=int, default=50,
                    help="MPPI rollouts per solve. Dominates runtime: a solve is "
                         "~1.8 s at 50, and an episode needs ~140 solves. Drop to "
                         "~20 for a quick smoke run.")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # The controller owns the model/data -- see SkyGripController. Rendering and
    # state must read the same MjData the MPPI is stepping.
    controller = SkyGripController(
        MODEL_PATH, MPPIParams(num_samples=args.samples, horizon=args.horizon))
    model, data = controller.model, controller.data
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    ik = ArmIK(model)

    solves_per_ep = len(PHASES) * FPS * (controller.CONTROL_HZ / FPS)
    print(f"~{solves_per_ep:.0f} MPPI solves/episode x {args.episodes} episodes. "
          f"At ~1.8 s/solve that is ~{solves_per_ep * args.episodes * 1.8 / 3600:.1f} h "
          f"of wall clock before rendering -- reduce --samples to trade accuracy for time.")

    # "names" is required on video features: feature_utils.py does names[2]
    # with no None guard, so omitting it raises TypeError at policy construction
    # -- and that same check is what drives the HWC->CHW shape flip.
    # timestamp/frame_index/episode_index/index/task_index are auto-populated;
    # passing them is rejected as extra features.
    features = {
        f"observation.images.{k}": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
        for k in CAMERAS
    }
    features["observation.state"] = {
        "dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES,
    }

    dataset = LeRobotDataset.create(
        repo_id=args.out_repo_id, fps=FPS, features=features, robot_type="skygrip",
    )

    for ep in range(args.episodes):
        for f in run_episode(model, data, renderer, controller, ik, rng, args.task):
            dataset.add_frame(f)          # v3.0: task lives INSIDE the frame dict
        dataset.save_episode()
        print(f"episode {ep + 1}/{args.episodes} ({len(PHASES) * FPS} frames)")

    # Without finalize() the parquet footer is never written and the dataset
    # is invalid -- not merely incomplete.
    dataset.finalize()
    print("done. dataset.push_to_hub() to upload.")


if __name__ == "__main__":
    main()
