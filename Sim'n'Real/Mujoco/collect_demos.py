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


WHY A PD FLIGHT CONTROLLER, NOT MPPI
------------------------------------
The grasp needs the jaws within ~10 mm laterally: the open jaws clear a 20 mm
block by only about 10 mm a side, so beyond that a jaw lands on the block instead
of around it and sweeps it off the pedestal.

MPPI could not deliver that. Measured steady-state station keeping over 30 s,
best of an 8-configuration sweep, was 106 mm RMS, and several tuned
configurations flipped the drone outright. The cascaded PD+I in pd_flight.py
holds 0.001 mm RMS. Episodes now track jaw-to-object error at 1-2 mm.

MPPI is still selectable with --flight mppi, and create_force_general.py also
gained a threaded batch rollout (7.6x, validated against the sequential path to
2.9e-12 relative cost error), but it is not the default.

The body, not the arm, corrects residual error. Swinging the arm to correct
flipped the drone during DESCEND; the PD tracks a shifted body setpoint instead.

VERIFIED END TO END
-------------------
Full flying episodes now complete the whole task: the block is grasped, lifted
~400 mm off the pedestal, carried, set back down and released, with the drone
airborne and attitude inside a few degrees throughout. Success is checked
against each episode's own randomised geometry (episode_succeeded) and FAILED
EPISODES ARE DISCARDED, not saved -- a failed demonstration teaches the policy
to fly the approach and then drop the block.

Observed success rate at the time of writing: 2/3 to 2/2 per batch. Re-check it
on a larger batch before assuming it holds across the full randomisation range.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer   # must be module level: importing it inside main() would
                       # rebind `mujoco` as a local and shadow this import

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from create_force_general import PureMPPIController, MPPIParams
from pd_flight import PDFlightController

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

# EXTEND exists because the arm and the body must never move at the same time.
# Joint_1 rotates about x and so does body roll, so arm reaction torque lands on
# the axis the flight controller is weakest about: an isolation test flipped the
# drone whenever the arm moved, and held attitude whenever it did not, in both
# hover and transit. Doing the whole travel->grasp swing (0.747 rad) in its own
# hovering phase means every other phase flies with a STATIC arm.
# Nothing after EXTEND changes the joints -- q_grasp is held all the way through
# PLACE. That is also the more stable pose: q_grasp is nearly straight down
# (0.32, 0.05) while q_travel is bent (0.27, -0.69), so the mass hangs below the
# thrust point instead of swinging out to one side.
PHASES = ["APPROACH", "EXTEND", "DESCEND", "GRASP",
          "LIFT", "TRANSPORT", "PLACE", "RELEASE"]

STATE_NAMES = (
    ["joint1", "joint2", "gripper"]
    + [f"drone_pos_{a}" for a in "xyz"]
    + [f"drone_quat_{a}" for a in "wxyz"]
    + [f"drone_linvel_{a}" for a in "xyz"]
    + [f"drone_angvel_{a}" for a in "xyz"]
)
ACTION_NAMES = ["drone_x", "drone_y", "drone_z", "joint1", "joint2", "gripper"]

# Gripper aperture, in metres of clamp travel. LOWER ctrl = MORE CLOSED.
#
# This was previously coded the other way round (OPEN=0.025, CLOSED=0.037) and it
# was wrong. Ray-casting straight through the grasp centre and rendering the jaws
# at a sweep of commands both show the blades TOGETHER at 0.000 and WIDEST at
# 0.037. The earlier "confirmation" from the slide-joint signs had the two clamps'
# sides swapped.
# Measured gap between the visible jaw faces (linear, 2 mm per 0.001 of command):
#     0.000 -> 15.7 mm   (closed; squeezes a 20 mm block by 4.3 mm)
#     0.005 -> 25.7 mm
#     0.013 -> 41.7 mm   (open; clears a 20 mm block by ~11 mm a side)
#     0.037 -> 89.7 mm   (fully open)
# Collision is done by the clamp MESHES themselves, so what is rendered is what
# grips. The invisible pad boxes that used to do this were removed: they had been
# placed on the wrong sides of their own parent blades, so they scissored the
# opposite way to the visible jaws and held the block 14 mm INSIDE the geometry
# on screen -- the block appeared to be cut into the gripper.
GRIPPER_OPEN = 0.013
GRIPPER_CLOSED = 0.000

# Phase advance. Two tolerances, because the phases have different needs:
#   TRANSIT  -- just needs to get the body into the neighbourhood.
#   PRECISE  -- used where the jaws must land on the object, and it is a HARD
#               geometric requirement, not a preference: the jaws open to a
#               40.7 mm gap around a ~20 mm post, so anything beyond ~10 mm of
#               lateral error puts a jaw through the object instead of around
#               it, and the object is swept off the pedestal.
# These were 0.10/0.05, sized for the MPPI, which could not resolve better than
# ~106 mm RMS. Under those tolerances a GRASP was allowed to proceed at 47 mm
# error and duly knocked the object to the floor. The PD controller settles to
# ~1 mm, so the tolerance is now set by what the GRIPPER needs rather than by
# what the flight controller could manage.
# HOLD_FRAMES requires the drone to STAY inside tolerance rather than clip through
# it at speed, which is what makes the grasp repeatable. MAX caps a phase that
# cannot converge so one bad episode does not stall a 50-episode run.
ARRIVE_TOL_TRANSIT = 0.05
ARRIVE_TOL_PRECISE = 0.012
# Setting down is a coarser job than picking up: the jaws only have to get
# the block near the surface before opening, and holding out for grasp
# precision costs the block (see the tol choice in the phase loop).
ARRIVE_TOL_PLACE = 0.035
HOLD_FRAMES = 8
MAX_FRAMES_PER_PHASE = int(4.0 * FPS)

# Per-phase overrides. These are CAPS, not durations: a phase exits as soon as it
# arrives and holds (HOLD_FRAMES), so a generous cap only matters for a phase that
# is still slewing. Overhead EXTEND finishes its 0.747 rad in ~2.5 s at 0.30 rad/s
# and exits then; the 12 s ceiling is there for the forward-reach reconfigures,
# which slew a larger angle at the gentle REACH_SLEW.
PHASE_FRAME_BUDGET = {
    "EXTEND": int(12.0 * FPS),
    # Forward-reach reconfigure phases. They slew the arm gently (REACH_SLEW) over
    # a large angle, so they need room; a phase that arrives sooner exits early on
    # HOLD_FRAMES, so a generous cap costs nothing on the fast overhead path.
    "REACH_OUT": int(12.0 * FPS),
    "DRAW_IN": int(12.0 * FPS),
    # PLACE both descends ~0.4 m AND translates to the place point while carrying
    # the block, so it needs longer than a plain transit. At 4 s it timed out
    # 168 mm short in y, and RELEASE then opened the jaws past the edge of the
    # pedestal and dropped the block on the floor -- an otherwise successful
    # pick-and-carry failing on the last phase.
    "PLACE": int(8.0 * FPS),
}


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

    CONTROL_HZ = 20.0        # replan rate; physics runs at 1 kHz

    # Jaw-position trim (see step()). Deliberately conservative: this authority
    # moves the whole aircraft, and the pedestal is close enough that an
    # over-eager correction lands the legs on it.
    # DISABLED (gain 0). This trim was written when the flight controller held
    # position to ~100 mm and the jaws had to be dragged onto the target. With
    # the PD controller the open-loop waypoints already put the jaws within
    # ~25 mm, and every version of the trim tried made things worse: it drove
    # the drone into the pedestal, and at lower gains still produced a 554 mm
    # runaway in -y during DESCEND. With it off, DESCEND arrives cleanly at
    # 25 mm. Re-enable only with a proper rate limit and a fix for that runaway.
    SERVO_ENGAGE = 0.10      # only trim once the jaws are within 100 mm
    SERVO_GAIN = 0.0         # per control step
    SERVO_LIMIT = 0.06       # total body offset the trim may command

    def __init__(self, model_path, params=None, flight="pd"):
        # "self.mppi" is now just "the flight controller" -- the PD controller
        # exposes the same mppi_step/apply_control/target_pos surface, so the FSM
        # below is unchanged by the swap.
        #
        # PD is the default because the difference is not marginal. Measured
        # steady-state station keeping over 30 s:
        #     MPPI, best of an 8-config sweep : 106 mm RMS (and several
        #                                       configurations flipped outright)
        #     PD + gravity feed-forward       : 0.001 mm RMS
        # The grasp needs ~10 mm, because the open jaws clear a 20 mm post by
        # only about 10 mm a side, so MPPI was an order of magnitude short of
        # being able to grasp at all. MPPI remains selectable for comparison.
        if flight == "pd":
            self.mppi = PDFlightController(model_path)
        else:
            self.mppi = PureMPPIController(model_path, params or MPPIParams())
        self.model = self.mppi.model
        self.data = self.mppi.data

        self.idx_arm = [self.model.actuator(n).id
                        for n in ("act_joint1", "act_joint2")]
        self.idx_gripper = self.model.actuator("act_gripper").id
        # Left jaw is driven directly, mirrored. See the actuator comment in
        # SkyGrip_full.xml: relying on the soft equality constraint to transmit
        # squeeze left the pads 20.4 mm apart on a 20 mm block (no grip at all).
        self.idx_gripper_left = self.model.actuator("act_gripper_left").id
        self.drone_target = np.array([0.0, 0.0, 1.5])
        self.joint_target = np.zeros(2)
        self.gripper_cmd = GRIPPER_OPEN

        self._ctrl_dt = 1.0 / self.CONTROL_HZ
        self._last_ctrl = -np.inf
        self._u = np.zeros(6)
        self._u[0] = self.mppi.nominal_hover_thrust
        self._servo_to = None
        self._ik = None
        self._q_cmd = np.zeros(2)   # slewed arm command actually sent to ctrl
        self._grip_cmd = GRIPPER_OPEN   # slewed gripper command (see step())
        self._servo_corr = np.zeros(3)
        # Overhead slew rate to restore between reach episodes -- captured here so a
        # --slew override (which sets the class attribute before construction) is
        # still respected. run_episode drops to REACH_SLEW for reach episodes.
        self._base_slew = self.ARM_SLEW_RATE

    # Max arm setpoint change per control step, rad. Joint_1 rotates about x and
    # so does body ROLL, which means arm reaction torque couples straight into the
    # axis the flight controller is weakest on: the arm actuators have +-6 N*m
    # while roll_torque has +-1 N*m. An unrestricted setpoint jump (e.g. the
    # 0 -> [0.89, -0.80] step at episode start) swings the arm in ~70 ms and rolls
    # the drone past 15 deg before MPPI can respond; it never recovers. Slewing
    # spreads the same motion over ~1 s so the reaction stays inside the body's
    # control authority.
    # NOTE: this is rad per SECOND, converted to a per-physics-step increment in
    # step(). Expressing it per call is a trap -- step() runs at the physics rate
    # (1 kHz), not the control rate, so a per-call cap gets applied ~33x per frame
    # and does nothing.
    # Swept at 25 samples / horizon 8 while the arm still moved DURING descent:
    # 0.6 and 0.3 flipped the drone, 0.12 stayed upright but never converged,
    # 0.05 held attitude but was so slow it never actually reached the grasp pose
    # inside the phase budget -- it bought stability by freezing the arm.
    # The arm was never the main destabiliser -- short MPPI lookahead was. Once
    # the horizon went from 8 (0.40 s) to 16 (0.80 s), a 30 s hover with the arm
    # slewing held attitude at every rate up to 0.30 rad/s and only failed at
    # 0.50. Measured max|roll| / final drift over 30 s:
    #     0.05 -> 15.3 deg / 0.352 m    0.15 -> 14.2 deg / 0.385 m
    #     0.30 -> 22.9 deg / 0.138 m    0.50 -> FLIPPED at t=0.62
    # 0.30 is chosen because it finishes the 0.747 rad swing in 2.5 s. That
    # matters beyond convenience: hover endurance is finite, so a shorter EXTEND
    # leaves more of the stability budget for the phases that need precision.
    # These outcomes are marginal and seed-dependent (an earlier 0.05 run failed
    # at t=25.9 where this one held), so treat the margin as thin.
    ARM_SLEW_RATE = 0.30

    # Max gripper command change per second. The gripper used to be written
    # straight to ctrl, so it SLAMMED from open (0.013) to closed (0.000) in a
    # single step. Overhead that is harmless -- the jaws are directly below the
    # body, so the closing-contact reaction is vertical and makes almost no body
    # moment. But on a FORWARD-REACH grasp the jaws are ~0.14 m out in front, so
    # the same impulsive contact acts on a long lever and torques the drone over:
    # measured GRASP flipping to roll 135-160 deg and knocking the block to the
    # floor even after a clean, millimetre-accurate descent. Closing over ~1.6 s
    # instead spreads the contact impulse so the attitude loop and the gravity
    # feed-forward keep up. It is also gentler on a 10 g low-friction block.
    # Per SECOND, converted to a per-physics-step increment in step() -- same unit
    # convention as ARM_SLEW_RATE.
    GRIPPER_SLEW_RATE = 0.008

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
        self._servo_corr = np.zeros(3)   # per-phase, never carried over
        self._ik = ik

    def reset_after_randomisation(self):
        """Re-seed the MPPI warm start after qpos is overwritten.

        u_init carries the previous episode's solution. Left alone, the first
        control steps of a new episode are optimising against a pose that no
        longer exists, which shows up as a lurch in the opening frames -- i.e.
        contaminating exactly the part of every episode the policy sees most.
        """
        if hasattr(self.mppi, "u_init"):          # MPPI warm start
            self.mppi.u_init[:] = 0.0
            self.mppi.u_init[:, 0] = self.mppi.nominal_hover_thrust
            self.mppi.u_prev = np.zeros(self.mppi.nu)
        else:                                      # PD integrator / reference
            # Carrying either across episodes would inject the previous episode's
            # accumulated error into the opening frames of the next one.
            self.mppi._i_pos = np.zeros(3)
            self.mppi._sp = None
        self._last_ctrl = -np.inf
        self._u = np.zeros(6)
        self._u[0] = self.mppi.nominal_hover_thrust
        # Start the slew from where the arm physically IS, not from the last
        # episode's command -- otherwise episode 2 opens with the same jump
        # this mechanism exists to prevent.
        self._q_cmd = np.array([self.data.qpos[7], self.data.qpos[8]])
        # Every episode opens in APPROACH with the jaws open; seed the slewed
        # gripper command to match so it does not carry over the last episode's
        # closed state and lurch open in the opening frames.
        self._grip_cmd = GRIPPER_OPEN

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
            # Close the loop on the MEASURED jaw position by moving the BODY.
            # The body reaches its waypoint to ~25 mm but the jaws still land
            # 28 mm sideways and 59 mm high, because the body->jaw offset is a
            # feed-forward from ik.solve_drop() and every error in it lands
            # straight on the grasp. grasp_site is observable, so servo on it.
            # The body is the right thing to move: swinging the ARM to correct
            # was measured flipping the drone during DESCEND (roll 155 deg),
            # whereas the PD controller tracks a shifted body setpoint to ~1 mm.
            if self._servo_to is not None:
                jaws = grasp_site_pos(self.model, self.data)
                # Integral correction on the FIXED waypoint. Writing
                # "target = current_position + error" instead makes the setpoint
                # sit a constant distance from the drone forever, which is a
                # velocity command, not a position one -- it flew the drone into
                # the ground at 0.15 m per control step. Accumulating against
                # drone_target converges: once the jaws reach servo_to the error
                # is zero and the correction stops growing.
                # Trim only, and only once the open-loop waypoint has already
                # brought the jaws close. Engaging it on the full 350 mm error at
                # the start of DESCEND saturated the clamp instantly, commanded
                # the body a quarter-metre off its waypoint, and flew it into the
                # pedestal. The waypoint flies the descent; this corrects the
                # residual body->jaw offset error, which is tens of millimetres.
                err = self._servo_to - jaws
                if np.linalg.norm(err) < self.SERVO_ENGAGE:
                    self._servo_corr = np.clip(
                        self._servo_corr + self.SERVO_GAIN * err,
                        -self.SERVO_LIMIT, self.SERVO_LIMIT)
                self.mppi.target_pos = self.drone_target + self._servo_corr

            if False and self._servo_to is not None and self._ik is not None:
                # Correct the vertical drop ONLY. The arm could also swing to fix
                # lateral error, but doing so throws its CoM sideways and stands up
                # a roll moment on the axis the body is weakest about -- the very
                # thing that was crashing the drone. Vertical is where the 5 mm
                # grasp window lives; lateral is the body's job.
                drop = float(self._servo_to[2] - self.data.qpos[2])
                q, off = self._ik.solve_drop(-drop if drop < 0 else drop)
                if abs(abs(off[2]) - abs(drop)) < 0.02:   # inside the envelope
                    self.joint_target = q
                    self.mppi.target_q = q

        self.mppi.apply_control(self._u)
        # Overwrite the arm/gripper channels AFTER apply_control: MPPI's solution
        # for them is discarded in favour of the commanded setpoint. MPPI re-plans
        # from measured state every step, so it absorbs the difference rather than
        # fighting it.
        #
        # The COMMANDED setpoint is slewed toward the target rather than applied
        # directly -- see ARM_SLEW. self._q_cmd is what the actuator actually sees.
        step_max = self.ARM_SLEW_RATE * self.model.opt.timestep
        delta = np.clip(self.joint_target - self._q_cmd, -step_max, step_max)
        self._q_cmd = self._q_cmd + delta
        # Tell MPPI what the arm is actually being commanded to do, so its rollouts
        # predict the real reaction torque instead of a sampled fiction.
        self.mppi.arm_cmd = self._q_cmd
        self.data.ctrl[self.idx_arm[0]] = self._q_cmd[0]
        self.data.ctrl[self.idx_arm[1]] = self._q_cmd[1]
        # Gripper is rate-limited the same way the arm is -- see GRIPPER_SLEW_RATE.
        # A slammed close flips a forward-reach grasp. self._grip_cmd is what the
        # actuator actually sees; the left jaw is the mirror of it.
        grip_step = self.GRIPPER_SLEW_RATE * self.model.opt.timestep
        self._grip_cmd += np.clip(self.gripper_cmd - self._grip_cmd,
                                  -grip_step, grip_step)
        self.data.ctrl[self.idx_gripper] = self._grip_cmd
        self.data.ctrl[self.idx_gripper_left] = -self._grip_cmd

    def gripper_settled(self):
        """True once the slewed gripper command has reached its target.

        The gripper now closes over ~1.6 s (GRIPPER_SLEW_RATE), but GRASP/RELEASE
        arrival is judged by JAW POSITION, which is satisfied the instant DESCEND
        finishes -- long before the jaws have actually closed. Without gating on
        this, GRASP would exit and LIFT would start lifting mid-close, dropping the
        block. So those phases wait for the grip to complete."""
        return abs(self._grip_cmd - self.gripper_cmd) < 1e-4

    def action(self):
        """Logged action = the COMMANDED setpoints, not MPPI's realised output.

        The VLA is being trained to emit setpoints for MPPI to track, so the
        supervision target is the setpoint. Logging MPPI's thrust/torque instead
        would train the policy to imitate the low-level controller.
        """
        return np.concatenate([self.drone_target, self.joint_target,
                               [self.gripper_cmd]]).astype(np.float32)


# Nominal camera poses, captured once so jitter is applied about the design pose
# rather than compounding episode to episode.
_CAM_NOMINAL = {}


def randomise_episode(model, data, rng):
    """Domain-randomise object, drone start pose, surface, cameras and lighting.

    Volume alone overfits: LeRobot's own guidance pairs "~50 episodes" with 5
    distinct object positions x 10 episodes, and documents 25 episodes as too
    few. Randomise per episode, not per batch of episodes.

    Factor priority follows the published evidence, which is consistent that
    SPATIAL factors matter far more than appearance ones:
      * Factor World (arXiv 2307.03659) measured, from a 91.7% baseline, camera
        position -45.9 pp and table texture -38.9 pp, but lighting only -8.4 and
        background -2.8.
      * LIBERO-Plus (arXiv 2510.13626) ranks camera viewpoint and robot initial
        state as the largest degradations across 10 VLAs including pi0.
    So camera pose and object/surface geometry get the wide ranges here, and
    lighting/colour get modest ones.

    Note the ranges are deliberately narrow rather than maximal. RCAN
    (arXiv 1812.07252) measured zero-shot grasp success of 37% under MILD
    randomisation but 35% medium and 33% heavy -- widening the ranges made it
    monotonically worse. Start narrow, widen only if real performance improves.
    """
    mujoco.mj_resetData(model, data)

    # --- pedestal: static body, so it moves by writing model.body_pos rather
    #     than qpos. Shifting the whole work surface (not just the object on it)
    #     stops the policy learning one fixed approach corridor.
    ped = model.body("pedestal").id
    ped_gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ped][0]
    ped_y = rng.uniform(0.28, 0.42)

    # A fraction of episodes have NO table at all -- the block sits on the ground.
    # This is the widest single change to the working height (0.40-0.60 m with a
    # pedestal, 0 m without), and it also removes the large coloured slab that
    # otherwise fills most of the wrist view, so the policy cannot rely on "the
    # object is on the bright rectangle". The pedestal is parked far below the
    # floor rather than deleted, since geometry cannot be added or removed from a
    # compiled model at run time.
    # Working HEIGHT varies widely, which is the single largest real-world factor
    # in the empirical study in arXiv 2603.22876 (~36.9% alone, above camera pose
    # at ~23.5%). Surface lands anywhere from 0.12 m to 0.60 m -- a 5x spread, so
    # the approach altitude and the whole descent differ episode to episode.
    #
    # Removing the table ENTIRELY was tried and is not reachable on this airframe:
    # the legs hang 0.20 m below the body and the arm reaches 0.19 m down, so a
    # floor pickup needs the body at ~0.23 m, leaving the legs 30 mm of clearance.
    # Both no-table episodes tested crashed on exactly that (drone down at
    # z=0.23/0.24). Raising the grasp with a taller block just moves the failure
    # to toppling, which is what the 70 mm blocks did. The height range below
    # gives the same variety without asking for the impossible.
    ped_half_h = rng.uniform(0.06, 0.30)
    model.geom_size[ped_gid, 2] = ped_half_h
    model.body_pos[ped] = [0.0, ped_y, ped_half_h]
    surface_z = 2.0 * ped_half_h

    # Surface appearance: Factor World ranked table texture second only to camera
    # pose (-38.9 pp). Without a texture library, vary the surface colour -- it is
    # the part of the wrist camera's view that dominates during the grasp.
    # Sampled as a warm grey rather than a free RGB: uniform(0.25,0.75) on all
    # three channels independently produces saturated magentas, limes and cyans,
    # which look nothing like a real bench and teach the policy to expect colours
    # it will never see. `lum` sets how light the surface is, `warm` how far it
    # leans from neutral grey towards brown/tan.
    lum = rng.uniform(0.26, 0.62)
    warm = rng.uniform(0.0, 0.30)
    model.geom_rgba[ped_gid, :3] = np.clip(
        [lum * (1.0 + warm), lum * (1.0 + 0.35 * warm), lum * (1.0 - 0.55 * warm)],
        0.05, 0.95)

    # --- object: on the pedestal surface, within its x extent, plus yaw so the
    #     grasp is not always axis-aligned (a fixed yaw teaches one approach only)
    bid = model.body("target_object").id
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    obj_gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]

    # --- object SIZE, bounded by what the gripper can actually hold. The pads
    #     close to a 16.7 mm gap, and a square post of side s at yaw t presents
    #     s*(|cos t| + |sin t|) across the jaws -- so size and yaw are coupled and
    #     cannot be sampled independently. GRIP_ENVELOPE is measured, not assumed
    #     (see grip_envelope.py); sampling outside it would generate episodes
    #     where the demonstrated grasp fails, which is worse than no episode.
    side = rng.uniform(*OBJ_SIDE_RANGE)
    depth = rng.uniform(*OBJ_DEPTH_RANGE)
    obj_half_h = rng.uniform(*OBJ_HALF_HEIGHT_RANGE)
    model.geom_size[obj_gid] = [side / 2, depth / 2, obj_half_h]
    # Widest presented width must still fit inside the closed gap plus the
    # squeeze the pads can absorb.
    # Yaw is limited by the BLIND axis, not the closing axis: rotating a
    # 20 x 12 mm post swings its wide dimension into the direction the housing
    # cannot clear. presented_depth(t) = side*|sin t| + depth*|cos t|, which at
    # only 15 deg already turns a 12 mm depth into ~17 mm -- past the measured
    # clearance limit. So this stays small; it is a property of the gripper, not
    # a randomisation preference.
    max_yaw = min(_max_safe_yaw(side), np.radians(10.0))
    obj = np.array([rng.uniform(-0.14, 0.14), ped_y, surface_z + obj_half_h])
    yaw = rng.uniform(-max_yaw, max_yaw)
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
    model.geom_rgba[obj_gid, :3] = rng.uniform(0.15, 0.9, size=3)

    # --- camera pose jitter. The highest-value factor in the literature and the
    #     one this script previously did not vary at all. Ranges follow NVIDIA's
    #     SO-101 sim-to-real recipe (x,y +-0.02 m, z +-0.01 m, +-0.05 rad).
    #     There is a platform-specific reason to keep these rather than trim
    #     them: UMI-on-Air (arXiv 2510.02614) injects noise matching a measured
    #     ~3 cm hover tracking error on real aerial manipulators, so a drone's
    #     effective camera pose is genuinely less repeatable than a fixed base.
    for cam_name in CAMERAS.values():
        cid = model.camera(cam_name).id
        if cid not in _CAM_NOMINAL:
            _CAM_NOMINAL[cid] = (model.cam_pos[cid].copy(),
                                 model.cam_quat[cid].copy())
        pos0, quat0 = _CAM_NOMINAL[cid]
        model.cam_pos[cid] = pos0 + rng.uniform([-0.02, -0.02, -0.01],
                                                [0.02, 0.02, 0.01])
        # Small-angle perturbation applied to the nominal orientation.
        ax = _unit(rng.normal(size=3))
        ang = rng.uniform(-0.05, 0.05)
        dq = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, dq, quat0)
        model.cam_quat[cid] = q

    mujoco.mj_forward(model, data)
    return obj


# Measured grip envelope -- see grip_envelope.py. Sampling outside this produces
# demonstrations where the grasp fails, so these bounds are a correctness
# constraint on the randomisation, not a style choice.
# Jaws aim this far behind the object centre in y, so the block sits in the
# clear part of the opening rather than against the housing.
GRASP_Y_OFFSET = 0.008

# How close to the requested place point the block must end up for the
# episode to count as a demonstration worth training on.
PLACE_TOL = 0.06

# How far the block is asked to travel, and how far out along the pedestal a
# place point may sit. The object spawns within +-0.14 and the pedestal is
# 0.20 half-length, so with a 0.16 limit the roomier side always has at least
# 0.16 m available -- the requested shift is therefore never truncated, and
# every episode is a genuine transport rather than a nudge.
PLACE_MIN_SHIFT = 0.10
PLACE_MAX_SHIFT = 0.16
PEDESTAL_X_LIMIT = 0.16

# Release height above the grasp height. Sets the block down instead of
# driving it into the surface -- see place_pt in build_plan.
PLACE_CLEARANCE = 0.006

# Forward-reach variety. A fraction of episodes reach the arm OUT to the object
# instead of descending vertically onto it, so the policy sees the manipulator
# extend forward and does not learn a single canned overhead approach (the user's
# request). The overhead grasp is kept as the majority because it is the most
# stable and highest-yield; forward reach is an additional variant, not a
# replacement. REACH_Y_RANGE is in body-frame y (the default straight-down pose
# hangs the jaws at ~-0.056); -0.09..-0.15 is the span proven flyable and
# leg-clear in reach_hover.py / leg_clear.py, with the mouth within ~10 deg of
# vertical so the jaws still descend squarely.
REACH_FRACTION = 0.5
REACH_Y_RANGE = (-0.15, -0.09)
# Arm slew rate for the forward-reach reconfigure phases. The overhead default is
# 0.30 rad/s, but the reach gesture swings the arm through a much larger angle, and
# at 0.30 the reaction torque of that fast swing shoves the body ~0.2 m off its
# hover before the phase finishes. 0.10 spreads the same motion over ~3x the time
# so the flight controller holds station throughout (measured: EXTEND then arrives
# at err ~0.01 m and dead level, vs timing out 0.1-0.2 m off at 0.30).
REACH_SLEW = 0.10


OBJ_SIDE_RANGE = (0.018, 0.022)     # along the jaws' closing axis
# Depth, along the gripper's BLIND axis. This is a hard clearance limit, not a
# style choice: the gripper housing occupies the column on the +y side of
# grasp_site, so a deep object is struck by the housing on the way down. Measured
# by descending onto posts of varying depth -- 20 mm and 16 mm were both swept
# off the pedestal (-0.5 m), 12 mm was left undisturbed with clean two-sided pad
# contact. Keep this well inside that limit.
OBJ_DEPTH_RANGE = (0.010, 0.013)
# Half-height. Short on purpose. The jaws grip 10 mm below the top face, so on a
# tall post that grip point sits far above the base and any nudge from the closing
# jaws has enormous leverage -- a 70 mm post toppled every time (it fell flat,
# a signature -29.0 mm drop, identical across every object width tried).
# Shorter blocks put the grip point close to the centre of mass.
# Half-height, squeezed between two opposing constraints:
#   TOO TALL  -> the jaws grip 10 mm below the top face, far above the base, so
#                the closing nudge has huge leverage. A 70 mm post toppled every
#                time (a signature -29.0 mm drop as it fell flat).
#   TOO SHORT -> the legs sit 0.01 m below the jaws, so leg-to-surface clearance
#                is (object_height - 0.020). A 28 mm block left ~8 mm and the
#                legs struck the pedestal during DESCEND and flipped the drone.
# 44-56 mm keeps 24-36 mm of leg clearance while staying well under the height
# that topples.
OBJ_HALF_HEIGHT_RANGE = (0.022, 0.028)
MAX_PRESENTED_WIDTH = 0.024      # pads close to 16.7 mm; this is the squeeze limit


def _max_safe_yaw(side):
    """Largest |yaw| whose presented width still fits the jaws.

    presented(t) = side * (|cos t| + |sin t|), which is monotonic on [0, 45 deg]
    and peaks at sqrt(2)*side. Solved rather than tabulated so it stays correct
    when OBJ_SIDE_RANGE changes.
    """
    if side * np.sqrt(2.0) <= MAX_PRESENTED_WIDTH:
        return np.pi / 4
    lo, hi = 0.0, np.pi / 4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if side * (np.cos(mid) + np.sin(mid)) <= MAX_PRESENTED_WIDTH:
            lo = mid
        else:
            hi = mid
    return lo


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
        # Lateral CoM offset of the arm for each grid pose. A 2-DOF arm reaching a
        # point in the y-z plane has multiple solutions, and they are NOT equally
        # good on a flying base: a folded "Z" pose can reach the same grasp point
        # while throwing the arm's mass ~50 mm to one side, which is a standing
        # ~0.1 N*m roll moment the flight controller then has to trim continuously.
        # Joint_1 rotates about x and so does roll, so this lands on the weakest
        # axis. Measured: the folded solution for a 0.169 m drop crashed the drone;
        # the straight-down solution for the SAME drop has zero offset.
        self._com_dy = np.array([self._com_offset(a, b) for a, b in self._grid])

        # How vertical the jaw MOUTH is at each pose: |local-z . world-z| of
        # grasp_site, 1.0 = pointing straight down. solve_reach uses it to keep
        # the jaws descending squarely onto an upright block even when the arm is
        # slung forward. Both joints rotate about x, so this only ever tilts the
        # mouth in the y-z plane -- the closing axis (x) stays horizontal.
        self._mouthz = np.array([self._mouth_align(a, b) for a, b in self._grid])

        self.x_offset = float(np.mean(self._offsets[:, 0]))
        spread = float(np.ptp(self._offsets[:, 0]))
        assert spread < 1e-6, f"arm moves in x by {spread:.4f} m -- x_offset is not constant"

    # Weight on lateral CoM offset when choosing between IK solutions, in metres
    # of apparent position error per metre of CoM offset. Large enough to prefer
    # the straight-hanging solution whenever both reach the target, small enough
    # not to trade away real reach accuracy (the grasp window is ~5 mm).
    COM_PENALTY = 0.30

    def _com_offset(self, j1, j2):
        """Lateral (y) offset of the arm subtree's CoM from the drone body."""
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[self._j1] = j1
        d.qpos[self._j2] = j2
        mujoco.mj_forward(self._m, d)
        return float(d.subtree_com[self._m.body("Link_1").id][1]
                     - d.xpos[self._m.body("base_link").id][1])

    def _offset(self, j1, j2):
        """grasp_site position relative to the drone body, in body coordinates."""
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0                      # identity quaternion
        d.qpos[self._j1] = j1
        d.qpos[self._j2] = j2
        mujoco.mj_kinematics(self._m, d)
        return d.site_xpos[self._sid] - d.xpos[self._m.body("base_link").id]

    def _mouth_align(self, j1, j2):
        """|local-z . world-z| of grasp_site: 1.0 = jaws pointing straight down."""
        d = self._d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[self._j1] = j1
        d.qpos[self._j2] = j2
        mujoco.mj_kinematics(self._m, d)
        return abs(float(d.site_xmat[self._sid].reshape(3, 3)[2, 2]))

    def solve_reach(self, drop, reach_y):
        """Pose that reaches FORWARD to a given lateral `reach_y` while keeping
        the jaw mouth as vertical as possible. Returns (angles, body->jaws offset).

        This is the deliberate opposite of solve_drop's CoM handling. solve_drop
        picks the straightest-hanging pose to keep the arm's mass on the gravity
        axis; reaching forward necessarily slings that mass sideways (up to
        ~0.11 m of lateral CoM at full reach). That was the configuration the old
        IK docstrings warn crashed the drone -- but that was under MPPI. The PD
        controller feeds the resulting gravity moment forward exactly, and a hover
        at full reach was re-measured holding station to <0.02 mm with zero tilt
        (reach_hover.py). So this ignores CoM and optimises for a vertical mouth
        instead, which is what lets the forward-reaching jaws still descend
        squarely onto an upright block.

        The requested `reach_y` is a target, not a guarantee: whatever grid pose
        is nearest is returned together with its TRUE offset, so the caller flying
        to (grasp_pt - offset) still lands the jaws exactly on the object.
        """
        want = np.array([reach_y, -abs(drop)])
        dyz = np.linalg.norm(self._offsets[:, 1:3] - want, axis=1)
        near = dyz < 0.015
        if not np.any(near):
            k = int(np.argmin(dyz))
            return self._grid[k].copy(), self._offsets[k].copy()
        idx = np.where(near)[0]
        k = int(idx[np.argmax(self._mouthz[idx])])
        return self._grid[k].copy(), self._offsets[k].copy()

    def solve_drop(self, drop):
        """Pose that achieves a vertical `drop` with the arm hanging as straight
        as possible. Returns (joint angles, full 3-D body->jaws offset).

        Do NOT ask for a specific lateral position here. At j1=j2=0 the jaws hang
        at y = -102 mm, so demanding y=0 forces the arm to swing forward, throwing
        its CoM ~45 mm sideways and standing up a ~0.085 N*m roll moment that the
        flight controller must trim forever. Joint_1 rotates about x and so does
        roll, so it loads the weakest axis and this alone crashed the drone.

        Instead: take the straightest pose for the required drop and let the
        CALLER move the drone to put those naturally-hanging jaws over the target
        -- the same treatment x_offset already gets, and the reason it works is
        the reason overhead grasping works at all: keep the mass on the gravity
        axis rather than slinging it out on a lever.
        """
        want_z = -abs(drop)
        errs = np.abs(self._offsets[:, 2] - want_z)
        feasible = errs < 0.01
        if not np.any(feasible):
            k = int(np.argmin(errs))
            return self._grid[k].copy(), self._offsets[k].copy()
        idx = np.where(feasible)[0]
        k = int(idx[np.argmin(np.abs(self._com_dy[idx]))])
        return self._grid[k].copy(), self._offsets[k].copy()

    def solve(self, target_offset):
        """Joint angles putting the jaws at `target_offset` from the body.

        Returns (angles, residual). CHECK THE RESIDUAL -- targets outside the
        arm's envelope return the closest reachable pose rather than failing, so
        a caller that ignores it will silently command a pose that does not
        reach the object.
        """
        t = np.asarray(target_offset, dtype=np.float64)

        def score(err, dy):
            """Reach error plus a penalty on slinging the arm's mass sideways."""
            return err + self.COM_PENALTY * abs(dy)

        errs = np.linalg.norm(self._offsets - t, axis=1)
        k = int(np.argmin(errs + self.COM_PENALTY * np.abs(self._com_dy)))
        best = self._grid[k].copy()
        best_err = float(errs[k])
        best_score = score(best_err, self._com_dy[k])

        span = (self._lim[0][1] - self._lim[0][0]) / len(np.unique(self._grid[:, 0]))
        for _ in range(self._refine_steps):
            improved = False
            for d1 in (-span, 0.0, span):
                for d2 in (-span, 0.0, span):
                    a = np.clip(best[0] + d1, *self._lim[0])
                    b = np.clip(best[1] + d2, *self._lim[1])
                    err = float(np.linalg.norm(self._offset(a, b) - t))
                    s = score(err, self._com_offset(a, b))
                    if s < best_score - 1e-9:
                        best = np.array([a, b])
                        best_err, best_score, improved = err, s, True
            span *= 0.5
            if not improved:
                span *= 0.5
        return best, best_err


def build_plan(ik, obj, place, obj_half_height, reach_y=None):
    """Per-phase (drone target, joint target, gripper) with joints from IK.

    The drone grasps from directly overhead. GRASP_DROP is the vertical distance
    from the body to the jaws, so the body flies to object_z + GRASP_DROP and the
    jaws land on the object. TRAVEL_DROP tucks the arm up between waypoints so
    the payload is not swinging at full extension during transit.

    reach_y: if given, the arm reaches FORWARD to hang the jaws at this body-frame
    y instead of the straight-down default (~-0.056). The body then flies further
    back to put the extended jaws over the object, so the arm visibly extends out
    rather than descending vertically. Everything downstream is unchanged: it uses
    the TRUE body->jaw offset returned for the pose, so the jaws still land exactly
    on the object regardless of how far forward the arm is reaching.
    """
    # GRASP_DROP is the NOMINAL body-to-jaw distance; the arm swings around it to
    # absorb body error. Deliberately well short of the 0.271 m envelope: the
    # headroom below it is how far the arm can reach down when the drone ends up
    # HIGH, and running out of it is an uncorrectable miss. At 0.23 there were
    # only 41 mm spare and a +80 mm body error left a 39 mm residual; 0.19 gives
    # ~81 mm, covering MPPI's error band in both directions. Costs leg clearance
    # (legs sit 0.20 m below the body) but still leaves ~100 mm over the pedestal.
    # GRASP_DROP is how far below the body the jaws hang, and it sets LEG
    # CLEARANCE: the legs sit 0.20 m below the body, so the gap between the legs
    # and the work surface is (GRASP_DROP - 0.20) plus the object's height.
    # Kept at 0.19 because that is the drop whose IK solution hangs the gripper
    # VERTICALLY: solve_drop(0.19) -> q=[0.32, 0.05], jaws pointing straight down.
    # Asking for 0.25 to buy leg clearance was a bad trade -- solve_drop(0.25)
    # returns q=[0.27, 0.59], swinging Joint_2 through 0.53 rad and tilting the
    # jaws off vertical, so they no longer descend squarely onto an upright block
    # (and the body->jaw lateral offset shifts 21 mm as well).
    # Leg clearance is bought with OBJECT HEIGHT instead: legs sit 0.01 m below
    # the jaws here, so clearance = object_height - 0.020.
    GRASP_DROP, TRAVEL_DROP = 0.19, 0.13
    CRUISE = 0.42                      # body height above the surface in transit

    # Put the jaw midpoint 15 mm BELOW the object's top face, so the contact pads
    # (24 mm tall, centred on grasp_site) close on the upper part of the block.
    # The previous +0.030 put the pads 30 mm ABOVE the top face and they shut on
    # empty air -- the object was never touched, which is exactly what the episode
    # logs showed (object z unchanged at 0.540). Grip verified over an 8-20 mm
    # band below the top face, so 15 mm sits in the middle of the window.
    # Derived from the half-height so it stays correct if the block is resized.
    # 10 mm below the top face. The clear column above grasp_site is only about
    # 12.6 mm tall before the housing intrudes, so letting the object protrude
    # 15 mm above the site left no margin and the housing clipped it on the way
    # down. 10 mm keeps the protrusion inside the clearance while still putting
    # the 24 mm pads on the upper part of the block.
    GRASP_RISE = obj_half_height - 0.010

    # Reaching poses hang shallower than the straight-down pose (the arm trades
    # depth for forward extension), so ask for a slightly smaller drop when
    # reaching -- 0.178 m is what the forward poses actually achieve while staying
    # within ~10 deg of vertical (see reach_probe2.py). The body altitude
    # self-adjusts through off_grasp, so this only affects how far down the jaws
    # are; leg clearance at these poses was measured at +12 mm or better.
    # The GRASP is ALWAYS overhead, even on a forward-reach episode. Gripping at
    # full forward extension is not stable on this airframe and it is not a tuning
    # miss: the instant the jaws firmly grip a block that is still resting on the
    # table, the drone is kinematically pinned to the ground through the arm at a
    # ~0.14 m horizontal lever, and the position/attitude loop diverges (measured
    # roll 0 -> 60 deg in ~0.5 s, dragging the gripped block with it; lifting
    # immediately does not break it in time). Overhead the same grip pins the
    # drone directly below its CoM -- zero lever, zero destabilising moment -- so
    # that is where the close happens. The forward reach is delivered instead as
    # an approach GESTURE with the gripper OPEN (no ground tether, provably
    # stable), which then draws in to this overhead pose before closing.
    q_grasp, off_grasp = ik.solve_drop(GRASP_DROP)
    q_travel, off_travel = ik.solve_drop(TRAVEL_DROP)

    # Body target = jaw target MINUS the body->jaw offset, in all three axes.
    # Using the full offset (not just x) is what lets the arm hang straight: the
    # drone flies to wherever puts the naturally-hanging jaws over the object,
    # rather than the arm reaching sideways to meet a body-centred target.
    at_obj = obj + np.array([0.0, GRASP_Y_OFFSET, GRASP_RISE]) - off_grasp
    at_place = place + np.array([0.0, GRASP_Y_OFFSET,
                                 GRASP_RISE + PLACE_CLEARANCE]) - off_grasp
    over_obj = obj + np.array([0.0, 0.0, CRUISE]) - off_travel
    # Same cruise waypoint, but for the phases flown with the arm at the grasp
    # pose. The body->jaw offset differs between the folded and grasp poses, so
    # reusing the off_travel version would put the now-extended jaws in the wrong
    # place.
    over_obj_g = obj + np.array([0.0, 0.0, CRUISE]) - off_grasp
    over_place_g = place + np.array([0.0, 0.0, CRUISE]) - off_grasp

    # Phases that must actually land on the object carry a world-space servo
    # point; transit phases do not need one.
    # Aim the jaws slightly BEHIND the object in y. The clear volume between the
    # jaws is not centred on grasp_site: the housing intrudes from about +5 mm on
    # the +y side, so an object centred on the site overlaps it by a millimetre or
    # two and gets shoved forwards on the way down (measured: the block was driven
    # +250 mm in +y and the reaction flipped the drone). Offsetting the aim point
    # puts the whole block inside the clear region.
    grasp_pt = obj + np.array([0.0, GRASP_Y_OFFSET, GRASP_RISE])
    # PLACE aims slightly HIGHER than GRASP. At the same height the block's base
    # reaches the surface before the jaws reach their target, so the drone spends
    # the whole phase pressing the block into the pedestal trying to close an
    # error it physically cannot -- PLACE timed out on its full budget in every
    # episode, successful or not, and that sustained fight is what destabilised
    # the failures (one crashed at roll 77 deg, another flung the block 0.7 m).
    # Releasing from a few millimetres up costs nothing: the block is 10 g.
    place_pt = place + np.array([0.0, GRASP_Y_OFFSET,
                                 GRASP_RISE + PLACE_CLEARANCE])

    # Return an ORDERED list of (name, body_target, joints, gripper, servo_point).
    # The head varies with approach style; everything from DESCEND on is identical
    # and always overhead.
    if reach_y is None:
        # Overhead: fly out with the arm folded, then reconfigure ONCE to the
        # grasp pose while hovering.
        head = [
            ("APPROACH", over_obj, q_travel, GRIPPER_OPEN, None),
            ("EXTEND",   over_obj, q_grasp,  GRIPPER_OPEN, None),
        ]
    else:
        # Forward reach: the arm extends OUT and reaches DOWN toward the object
        # with the gripper open (visibly a forward reach, and stable because there
        # is no grip and so no ground tether), then DRAWS IN -- retracts to the
        # vertical grasp pose over the object -- before the overhead grasp. The
        # reaching jaws come to REACH_OUT_RISE above the grip point: low enough to
        # read as reaching for the object, high enough that drawing the arm back up
        # and over clears it.
        REACH_DROP = 0.178
        REACH_OUT_RISE = GRASP_RISE + 0.14
        q_reach, off_reach = ik.solve_reach(REACH_DROP, reach_y)
        reach_out = obj + np.array([0.0, GRASP_Y_OFFSET, REACH_OUT_RISE]) - off_reach
        head = [
            ("APPROACH", over_obj,   q_travel, GRIPPER_OPEN, None),
            ("REACH_OUT", reach_out, q_reach,  GRIPPER_OPEN, None),
            ("DRAW_IN",  over_obj_g, q_grasp,  GRIPPER_OPEN, None),
        ]
    return head + [
        # Everything below flies with a static arm at the overhead grasp pose.
        ("DESCEND",   at_obj,       q_grasp, GRIPPER_OPEN,   grasp_pt),
        ("GRASP",     at_obj,       q_grasp, GRIPPER_CLOSED, grasp_pt),
        ("LIFT",      over_obj_g,   q_grasp, GRIPPER_CLOSED, None),
        ("TRANSPORT", over_place_g, q_grasp, GRIPPER_CLOSED, None),
        ("PLACE",     at_place,     q_grasp, GRIPPER_CLOSED, place_pt),
        ("RELEASE",   at_place,     q_grasp, GRIPPER_OPEN,   place_pt),
    ]


def episode_succeeded(model, data, info):
    """Did the block actually get picked up and put down where it was asked?

    The old check was `obj_z > 0.55`, hardcoded to the original fixed 0.50 m
    pedestal and 40 mm half-height. Once pedestal height and object size became
    randomised that number meant nothing, and it reported a genuinely successful
    pick-and-place as "not lifted". Success is defined against THIS episode's
    geometry instead.
    """
    oadr = model.jnt_qposadr[model.body_jntadr[model.body("target_object").id]]
    obj = data.qpos[oadr:oadr + 3]
    place = info["place"]
    surface = info["surface_z"]
    half_h = info["obj_half_height"]

    # Crash = the body has fallen to or below the work surface. This used to be a
    # hard 0.35 m, which was fine while every pedestal put the grasp near 0.75 m,
    # but a forward-reach grasp hangs the jaws ~20 mm shallower and so flies the
    # body lower, and on the shortest pedestals (surface 0.12 m) it legitimately
    # ends up hovering at ~0.33 m -- below 0.35, so a perfectly good pick was
    # reported as a crash. Judging against the surface keeps the check meaningful
    # at every working height: a real crash tumbles the drone to the floor, well
    # below the surface, while a low-but-controlled hover stays above it.
    if data.qpos[2] < surface + 0.05:
        return False, f"drone crashed (z={data.qpos[2]:.2f}, surface={surface:.2f})"
    # On the floor means it was dropped, not placed.
    if obj[2] < surface - 0.02:
        return False, f"object on the floor (z={obj[2]:.3f})"
    # Resting on the surface, allowing for the block having settled or tipped.
    if abs(obj[2] - (surface + half_h)) > 2.5 * half_h:
        return False, f"object not resting on the surface (z={obj[2]:.3f})"
    d = float(np.linalg.norm(obj[0:2] - place[0:2]))
    if d > PLACE_TOL:
        return False, f"object {d*1000:.0f} mm from the place target"
    moved = float(np.linalg.norm(obj[0:2] - info["obj_start"][0:2]))
    return True, (f"placed {d*1000:.0f} mm from target, "
                  f"moved {moved*1000:.0f} mm from start")


def run_episode(model, data, renderer, controller, ik, rng, task_text,
                viewer=None, verbose=False):
    obj = randomise_episode(model, data, rng)
    controller.reset_after_randomisation()

    # Place target: same surface, offset along the pedestal's long axis.
    #
    # The direction is chosen by which side has ROOM, not at random. Sampling the
    # sign and then clipping to the pedestal (the previous approach) silently
    # collapsed the displacement whenever the block spawned near an edge and the
    # sign pointed outward: observed episodes that moved the block only 21 mm and
    # 31 mm where others moved 120 mm. Those still pass the success check -- the
    # block does reach the requested point -- but they are near-no-op
    # demonstrations, and training a VLA on "pick and place" examples where the
    # object barely moves teaches an inconsistent notion of the task.
    reach = rng.uniform(PLACE_MIN_SHIFT, PLACE_MAX_SHIFT)
    room_pos = PEDESTAL_X_LIMIT - obj[0]      # room to the +x side
    room_neg = obj[0] + PEDESTAL_X_LIMIT      # room to the -x side
    if room_pos >= reach and room_neg >= reach:
        direction = float(rng.choice([-1.0, 1.0]))       # both fit: free choice
    else:
        direction = 1.0 if room_pos > room_neg else -1.0  # take the roomier side
    reach = min(reach, room_pos if direction > 0 else room_neg)
    place = np.array([obj[0] + direction * reach, obj[1], obj[2]])
    obj_half_height = model.geom_size[
        [g for g in range(model.ngeom)
         if model.geom_bodyid[g] == model.body('target_object').id][0]][2]
    # Per-episode approach style: mostly overhead, some forward-reach (see
    # REACH_FRACTION). reach_y=None is the vertical descent; a value reaches out.
    reach_y = None
    if rng.random() < REACH_FRACTION:
        reach_y = float(rng.uniform(*REACH_Y_RANGE))
    # The forward-reach gesture swings the arm gently; the overhead path keeps its
    # faster slew. Reset every episode so a reach does not slow the next overhead.
    controller.ARM_SLEW_RATE = REACH_SLEW if reach_y is not None else controller._base_slew
    plan = build_plan(ik, obj, place, obj_half_height, reach_y=reach_y)
    info = {"place": place.copy(), "obj_start": obj.copy(),
            "obj_half_height": obj_half_height,
            "reach_y": reach_y,
            "surface_z": float(obj[2] - obj_half_height)}
    if verbose:
        style = f"FORWARD-REACH y={reach_y:+.3f}" if reach_y is not None else "OVERHEAD"
        print(f"  approach style: {style}")

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

    for phase, drone_xyz, joints, grip, servo_to in plan:
        controller.set_targets(drone_xyz, joints, grip, servo_to=servo_to, ik=ik)

        # Phases advance on ARRIVAL, not on a timer. A fixed 1 s per phase cuts
        # the drone off mid-transit -- APPROACH alone can be a 1 m descent, which
        # MPPI cannot complete in a second -- so every later phase then starts
        # from the wrong place and the grasp misses for reasons that have nothing
        # to do with the IK. MIN keeps a few frames even when already on target
        # (so GRASP/RELEASE still show the jaws moving); MAX stops a phase that
        # cannot converge from hanging the whole run.
        held = 0
        budget = PHASE_FRAME_BUDGET.get(phase, MAX_FRAMES_PER_PHASE)
        for f_i in range(budget):
            for _ in range(substeps):
                controller.step()
                mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            # On precise phases, judge arrival by where the JAWS are, not where
            # the body is. The body setpoint is being actively shifted by the
            # servo above, so body-vs-waypoint error no longer means anything
            # there -- and the jaws are what has to be on the object.
            if servo_to is not None:
                # Setting the block DOWN does not need grasp precision. Holding
                # PLACE to the 12 mm grasp tolerance meant it usually could not
                # arrive, ran its full 8 s budget, and spent that time hovering
                # and nudging the block against the surface until it popped out
                # of the jaws -- the block then landed wherever luck put it, on
                # the pedestal in the successes and on the floor in the failures.
                # Episodes where PLACE arrived quickly (~62 frames) succeeded;
                # every episode where it timed out lost the block.
                tol = (ARRIVE_TOL_PLACE if phase in ("PLACE", "RELEASE")
                       else ARRIVE_TOL_PRECISE)
                arrived = np.linalg.norm(
                    grasp_site_pos(model, data) - servo_to) < tol
            else:
                tol = ARRIVE_TOL_TRANSIT
                arrived = np.linalg.norm(data.qpos[0:3] - drone_xyz) < tol
            # An arm-reconfiguration phase is finished when the ARM gets there.
            # Judging it by body position alone would let it exit the moment the
            # hover settles, with the joints still mid-slew -- which is precisely
            # how the jaws ended up 36 mm short of the grasp pose. EXTEND (overhead)
            # and REACH_OUT/DRAW_IN (forward-reach gesture) all slew the arm.
            if phase in ("EXTEND", "REACH_OUT", "DRAW_IN"):
                arrived = arrived and np.max(
                    np.abs(data.qpos[7:9] - joints)) < 0.02
            # GRASP/RELEASE actuate the gripper, which now slews shut/open over
            # ~1.6 s; the phase is not done until the jaws have finished moving,
            # or LIFT starts before the block is held (see gripper_settled).
            if phase in ("GRASP", "RELEASE"):
                arrived = arrived and controller.gripper_settled()
            if arrived:
                held += 1
            else:
                held = 0

            frame = {
                "observation.state": get_state(model, data),
                "action": controller.action(),
                "task": task_text,
            }
            # Rendering two 480x640 cameras per frame dominates runtime, and a
            # --no-save run discards them. Skip it there so visual debugging with
            # --view stays responsive.
            if renderer is not None:
                for key, cam in CAMERAS.items():
                    renderer.update_scene(data, camera=cam)
                    frame[f"observation.images.{key}"] = renderer.render().copy()
            frames.append(frame)

            if held >= HOLD_FRAMES:
                break

        if verbose:
            err = np.linalg.norm(data.qpos[0:3] - drone_xyz)
            gs = grasp_site_pos(model, data)
            oadr = model.jnt_qposadr[
                model.body_jntadr[model.body("target_object").id]]
            op = data.qpos[oadr:oadr + 3]
            # Jaw-to-object offset is what actually decides the grasp, so print
            # that rather than only the body's error against its own waypoint --
            # the body can be perfectly on target while the jaws miss the object.
            dxy = np.linalg.norm(gs[0:2] - op[0:2])
            print(f"    {phase:<10} {f_i + 1:3d} frames "
                  f"{'ARRIVED' if held >= HOLD_FRAMES else 'timeout'} | "
                  f"body {np.round(data.qpos[0:3], 3)} err={err:.3f} | "
                  f"jaws z={gs[2]:.3f} | obj {np.round(op, 3)} "
                  f"jaw-obj dxy={dxy*1000:.0f}mm dz={(gs[2]-op[2])*1000:+.0f}mm | "
                  f"roll/pitch={np.degrees(_rp(data)):.0f}/{np.degrees(_rp(data,1)):.0f} deg")

    return frames, info


def _rp(data, idx=0):
    """Roll (idx 0) or pitch (idx 1) from the drone quaternion."""
    w, x, y, z = data.qpos[3:7]
    if idx == 0:
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))


def pick_rgb_encoder():
    """Choose a video codec that actually exists in this environment.

    LeRobot defaults to libsvtav1, which is not in the pyav wheel on Windows --
    RGBEncoderConfig() then raises "Unsupported video codec" before a single
    frame is written. Rather than hardcode a codec (which would break on the
    machine that DOES have svtav1), ask pyav what it has and take the first
    preference that is present.

    h264 is preferred over the AV1 codecs on purpose: torchcodec is unavailable
    on Windows so decoding falls back to pyav, and h264 decodes considerably
    faster there. Training reads these videos far more often than we write them.
    """
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import detect_available_encoders_pyav

    preference = ["h264", "libsvtav1", "hevc", "libaom-av1"]
    available = detect_available_encoders_pyav(preference)
    for codec in preference:
        if codec in available:
            if codec != preference[0]:
                print(f"note: falling back to '{codec}' (h264 unavailable here)")
            return RGBEncoderConfig(vcodec=codec)
    raise RuntimeError(
        f"pyav has none of {preference}. Either install a fuller ffmpeg build, or "
        f"pass use_videos=False to LeRobotDataset.create to store PNG frames instead."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out_repo_id", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # Held CONSTANT across episodes, which is what the evidence supports for
    # single-task fine-tuning: LeRobot's own rollout instructions say to reuse
    # the recording task string verbatim, and LIBERO-Plus (arXiv 2510.13626)
    # found VLAs largely ignore the instruction semantically while remaining
    # sensitive to phrasing as a distribution shift -- so varying it buys little
    # and risks a mismatch at inference.
    # No longer says "red cube": the object colour is randomised per episode and
    # it is a rectangular block, so the old string was simply false for most of
    # the data. Short and action-verb-first, per the SmolVLA dataset guidance.
    ap.add_argument("--task", type=str,
                    default="Pick up the block and place it")
    ap.add_argument("--samples", type=int, default=50,
                    help="MPPI rollouts per solve. Dominates runtime: a solve is "
                         "~1.8 s at 50, and an episode needs ~140 solves. Drop to "
                         "~20 for a quick smoke run.")
    # Horizon 16 = 0.80 s of lookahead. Measured over a 20 s frozen-arm hover:
    # horizon 8 (0.40 s) diverged at t=2.3, horizon 16 held station (max roll
    # 26 deg, drift 0.21 m), horizon 25 diverged at t=0.5. Short lookahead, not
    # arm reaction torque, is what had been flipping the drone -- every phase
    # that ever reported ARRIVED was <=4 s, inside the window before divergence.
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--max_attempts", type=int, default=None,
                    help="Cap on total attempts when retrying to reach the "
                         "requested number of SUCCESSFUL episodes. Defaults "
                         "to 4x --episodes.")
    ap.add_argument("--flight", choices=["pd", "mppi"], default="pd",
                    help="Flight controller. 'pd' is the cascaded PD+I with "
                         "gravity feed-forward (0.001 mm RMS station keeping); "
                         "'mppi' is the original sampler (106 mm at best, and "
                         "flips on some tunings). The grasp needs ~10 mm.")
    ap.add_argument("--slew", type=float, default=None,
                    help="Override ARM_SLEW_RATE (rad/s). The arm's reaction "
                         "torque couples into body roll, so how FAST the arm "
                         "reconfigures is a stability parameter, not just a "
                         "cosmetic one. Lower = gentler = slower phases.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print a per-phase summary: frames used, whether the phase "
                         "arrived or timed out, body position/error, jaw height and "
                         "attitude. This is what tells you WHERE an episode fails.")
    ap.add_argument("--view", action="store_true",
                    help="Open the MuJoCo viewer and watch the episode run.")
    ap.add_argument("--no-save", action="store_true",
                    help="Skip LeRobot entirely -- fly the FSM and render nothing "
                         "to disk. Pair with --view for pure visual debugging; it "
                         "also sidesteps any video-encoding problems in the env.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete an existing dataset at this repo_id first. "
                         "LeRobotDataset.create() calls mkdir(exist_ok=False), so "
                         "a re-run otherwise dies with FileExistsError.")
    args = ap.parse_args()

    if not args.no_save and not args.out_repo_id:
        ap.error("--out_repo_id is required unless --no-save is given")

    if args.overwrite and not args.no_save:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root = Path(HF_LEROBOT_HOME) / args.out_repo_id
        if root.exists():
            shutil.rmtree(root)
            print(f"removed existing dataset at {root}")

    rng = np.random.default_rng(args.seed)

    # The controller owns the model/data -- see SkyGripController. Rendering and
    # state must read the same MjData the MPPI is stepping.
    if args.slew is not None:
        SkyGripController.ARM_SLEW_RATE = args.slew
    controller = SkyGripController(
        MODEL_PATH, MPPIParams(num_samples=args.samples, horizon=args.horizon),
        flight=args.flight)
    model, data = controller.model, controller.data
    # No renderer when nothing is being saved -- see run_episode.
    renderer = None if args.no_save else mujoco.Renderer(model, height=IMG_H, width=IMG_W)
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

    dataset = None
    if not args.no_save:
        dataset = LeRobotDataset.create(
            repo_id=args.out_repo_id, fps=FPS, features=features,
            robot_type="skygrip", rgb_encoder=pick_rgb_encoder(),
        )

    viewer = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

    try:
        # Retry until the requested number of SUCCESSFUL episodes is banked.
        # A failed demonstration is not neutral training data -- it teaches the
        # policy to fly the approach and then drop the block, so failures are
        # discarded rather than saved. Attempts are capped so a regression that
        # makes every episode fail stops instead of looping forever.
        saved = attempts = 0
        max_attempts = args.max_attempts or (args.episodes * 4)
        while saved < args.episodes and attempts < max_attempts:
            attempts += 1
            frames, info = run_episode(model, data, renderer, controller, ik,
                                       rng, args.task, viewer=viewer,
                                       verbose=args.verbose)
            ok, why = episode_succeeded(model, data, info)
            if not ok:
                print(f"  attempt {attempts}: DISCARDED -- {why}")
                continue
            saved += 1
            if dataset is not None:
                for f in frames:
                    dataset.add_frame(f)   # v3.0: task lives INSIDE the frame dict
                # parallel_encoding=False: with the default (True), LeRobot's
                # per-episode stats pass races the encoder, which has already
                # consumed and deleted the frame PNGs it is trying to sample --
                # FileNotFoundError on frame-000000.png three episodes into a
                # 50-episode run. Serialising costs some wall clock and makes
                # the run survivable.
                dataset.save_episode(parallel_encoding=False)
            style = "reach" if info.get("reach_y") is not None else "overhead"
            print(f"episode {saved}/{args.episodes} (attempt {attempts}): "
                  f"{len(frames)} frames | SUCCESS | {style} | {why}")
        if saved < args.episodes:
            print(f"WARNING: only {saved}/{args.episodes} succeeded in "
                  f"{attempts} attempts -- dataset is short.")
        else:
            print(f"banked {saved} successful episodes in {attempts} attempts "
                  f"({100.0*saved/attempts:.0f}% success rate)")

            # Report the outcome rather than just the frame count -- "it ran" and
            # "it worked" are different things, and only the second one matters.
    finally:
        if viewer is not None:
            viewer.close()

    if dataset is not None:
        # Without finalize() the parquet footer is never written and the dataset
        # is invalid -- not merely incomplete.
        dataset.finalize()
        print("done. dataset.push_to_hub() to upload.")
    else:
        print("done (--no-save: nothing written).")


if __name__ == "__main__":
    main()
