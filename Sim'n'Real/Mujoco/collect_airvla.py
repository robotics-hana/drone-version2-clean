"""AirVLA-recreation data collector (Reports/AIRVLA_RECREATION.md, Phase B).

Collects scripted-expert demonstrations in the scanned-lab scene
(SkyGrip_airvla.xml) with the paper's interface:

  observations  three RGB cameras at 256x256, logged at 10 Hz
                  camera1 = wrist_cam   (downward, gripper in frame)
                  camera2 = scene_cam   (forward onboard)
                  camera3 = overview_cam (static external)
                proprio [x y z qw qx qy qz aperture j1 j2]
  actions       7-D per-step WORLD deltas of the commanded setpoint at
                10 Hz: [dx dy dz droll dpitch dyaw grip]; roll/pitch are
                identically 0 (underactuated), yaw held 0 in this scene,
                grip is the commanded aperture fraction (1 = open).
                The expert emits a RAMPED setpoint (capped step per tick),
                so deltas are smooth pilot-like commands, never waypoint
                jumps.

Tasks (prompts follow the paper's phrasing):
  manip       "pick up the mustard bottle and put it in the blue bin"
  nav         "fly through the gate and hover over the mustard bottle"
  corrective  nav with perturbed starts + a waypoint near a random gate
              extremity (the paper's splat-synthesised recovery episodes,
              realised directly in sim)

Embodiment rules inherited from the probes (doc D5/D8): 200 g bottle
variant is the default payload; cap grasp aims cap_top - 4 mm on the cap
AXIS; staged descent with settled jaw-residual correction; close gated on
measured alignment; arm articulates travel -> grasp -> travel.

Run (review sample):
    python collect_airvla.py --episodes 15 --out_repo_id hanapasta/airvla_sample15
"""
import argparse

import numpy as np
import mujoco
from PIL import Image

import collect_demos as C

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MODEL = "SkyGrip_airvla.xml"
FPS = 10                      # log/control tick rate (= paper action rate)
IMG = 512                     # STORED resolution (D33: store above
                              # the model input, downsample in the
                              # training pipeline -- crop margin +
                              # headroom for higher-res models; the
                              # paper stored 256 and trained at 224,
                              # we store 512 and train at 224: same
                              # training inputs, more future-proof)
CAMS = {"camera1": "wrist_cam", "camera2": "scene_cam",
        "camera3": "lab_external"}   # full-workspace external (review fix)

SP_STEP = 0.035               # max setpoint move per tick, m (0.35 m/s)
SP_STEP_CARRY = 0.012         # gentle carry: 0.12 m/s while holding an object
YAW_STEP = 0.04               # rad/tick at 10 Hz = 0.4 rad/s turn rate
YAW_STEP_CARRY = 0.02         # gentle loaded turn: the yaw-in-place
                              # lead compensation must counter-rotate
                              # the setpoint at (yaw rate x lead), and
                              # at 0.4 rad/s x 0.28 m that outruns the
                              # slew taper (~6 cm lag -> 0.17 m body
                              # drift, measured); at 0.2 rad/s the arc
                              # tracks and the body holds station
                              # -- at full speed the pitch to accelerate swings
                              # the bottle on its cap pinch and rotational slip
                              # sheds it mid-transport (measured: every carry
                              # dropped along the mat->bin path)
SP_STEP_FINAL = 0.008         # creep for the last approach leg: the
                              # loaded PD glides ~0.3 m past an
                              # abruptly-stopped setpoint (measured);
                              # arriving at 0.05 m/s kills the glide
GRIP_STEP = 0.15              # max grip-fraction change per tick (~0.7 s stroke)

BOTTLE_MASS = 0.060           # HALF-FULL bottle (review 2026-08-19: the
                              # empty 30 g bottle kept getting knocked
                              # over -- doubling the mass doubles tip
                              # resistance and knock inertia, still well
                              # under the 100 g hardware ceiling, and
                              # keeps a sag regime distinct from the
                              # 100 g weight). Was 0.030. The 200 g sag-regime
                              # payload (D5) moves to a purpose-built
                              # calibration-weight object -- cap-grasping a
                              # tall bottle at >=100 g is a tipping-lever
                              # problem (filmed; see doc D9). Review sample
                              # and light-variation episodes use 30 g.
CAP_TOP = 0.191               # bottle on the physics floor (D8a)
AIM_Z = CAP_TOP - 0.004       # pinch at the VERY TOP of the cap (review
                              # 2026-08-19: "the rest of the bottle is too
                              # wide"). The weld carries the load, so the
                              # -7 mm deep pinch of the ratchet era
                              # (history below) is no longer needed:
                              # (old note) the shallow
                              # grip's ~12 mm of pad overlap let the round
                              # cap RATCHET down out of the pinch under
                              # carry oscillations (filmed: clean mid-carry
                              # release at ~14 s, 100x static margin --
                              # creep, not slip). -7 mm nearly doubles the
                              # overlap and keeps 3 mm of housing margin.
# Side grasp for the upright bottle (user proposal 2026-08-19, probe
# 8/8 vs the overhead grasp's ~50-60%): Joint_2 rotated so the jaw
# mouth faces forward-horizontal; the drone approaches the cap from
# behind at constant altitude and slides the mouth over it. No vertical
# motion near the object = the whole descent-knock family (housing
# sweep, loaded-trim y-swing, descent overshoot) is bypassed.
Q_SIDE = np.array([-0.747, -0.587])
OFF_SIDE = np.array([-0.007, -0.106, -0.065])
SIDE_ENTRY = np.array([0.0, 0.96, 0.24])   # reverse of the mouth axis
SIDE_STANDOFF = 0.14

RING_R = 0.45                 # D40 lateral grasp: orbit/approach ring
PLATE_TOP = 0.39              # stand plate top (incl. the 40 mm mat)
MAT_TOP = 0.04                # 40 mm gym-mat slab; everything task-
                              # related stands on it (D31)
GATE_LEFT, GATE_RIGHT = (-0.7, -0.6), (0.7, -0.6)
BIN_XY = np.array([1.4, 1.6])

# Prompt templates: {obj} is the episode's task object. Object variety
# (user request 2026-08-19) so the policy generalises past a single item;
# the object set is everything the hardware can actually handle -- the
# 100 g calibration weight and the empty (30 g) mustard bottle, the only
# graspable-and-stable survivor of prepare_objects.py's 22-YCB screen.
PROMPT_MANIP = "pick up the {obj} and put it in the wooden box"
PROMPT_NAV = "fly through the gate and hover over the {obj}"
WEIGHT_AIM_Z = 0.057          # stem top (0.062) - 5 mm, floor-resting weight
# The paper's third task (compositional "fly through the gate and put the
# {obj} in the box") is deliberately NOT collected: it is the held-out
# eval prompt, exactly as in the paper (doc section 4).

STATE_NAMES = (["x", "y", "z", "qw", "qx", "qy", "qz",
                "gripper_aperture", "joint1", "joint2"])
# Scene-state sidecar (D33, splat future-proofing): full poses of the
# scene. Deliberately NOT under the observation.* prefix: LeRobot's
# dataset_to_policy_features types every observation.* key as a STATE
# input and pi0 pads state to 32 dims -- a prefixed sidecar would
# silently leak into the policy. Unprefixed keys are skipped.
# dynamic scene, logged per frame but NEVER fed to the policy -- exists
# so a later Gaussian-splat re-render can reconstruct every frame's
# visuals without replaying physics.
SCENE_NAMES = (["task_x", "task_y", "task_z", "task_qw", "task_qx",
                "task_qy", "task_qz", "other_x", "other_y", "other_z",
                "other_qw", "other_qx", "other_qy", "other_qz",
                "bin_x", "bin_y", "bin_yaw",
                "bin2_x", "bin2_y", "bin2_yaw"])
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


# ---------------------------------------------------------------------------
# expert
# ---------------------------------------------------------------------------
class Expert:
    """Ramped-setpoint scripted pilot over the PD platform.

    Owns the commanded setpoint `sp` and grip fraction `grip`; tick() slews
    them toward the active goal and returns the recorded 7-D delta action.
    The PLATFORM (SkyGripController) separately owns the arm pose.
    """

    def __init__(self, ctrl, ik):
        self.ctrl, self.ik = ctrl, ik
        # solve_drop poses, UNCHANGED from the validated 15/15 sample run.
        # (2026-08-18: a re-selection of these poses by jaw orientation was
        # tried and REVERTED -- changing q_grasp changes the pad-stem
        # contact geometry the close/seat physics was validated on, and the
        # traced result was a ghosted close (0.3 mm aperture), a 0.34 m
        # airframe shove at seat, and weld-vs-floor solver drag that pinned
        # the drone at 0.15 m altitude. The camera1 view regression the
        # re-selection tried to fix was actually the wrist_cam XML edit,
        # restored separately in SkyGrip_core.xml.)
        # FORWARD-reaching poses (user request 2026-08-19): solve_drop's
        # CoM-optimal solutions reach BACKWARD under the body, which reads
        # as a weird gripper angle on camera. The arm is symmetric about
        # zero (both joints axis="1 0 0"), so the mirrored joint solution
        # (-j1, -j2) reaches forward at the same drop. Offsets are looked
        # up from the FK grid at the mirrored angles, never sign-flipped.
        self.q_travel, self.off_travel = self._mirrored(ik, 0.16)
        self.q_grasp, self.off_grasp = self._mirrored(ik, 0.19)
        # Carry pose (review 2026-08-19: "the object looks like it is
        # floating in space"): the scene camera's frustum cannot see the
        # region under the body where the grasp poses hang, so during the
        # carry the arm extends to its furthest FORWARD reach at a shallow
        # drop -- gripper and welded object sit mid-frame in camera2, the
        # object visibly hangs from the arm, and the pd_flight arm-CoM
        # feed-forward absorbs the shifted trim.
        self.q_carry, self.off_carry = self._forward_carry(ik)
        # D41 TUCK pose: minimal horizontal jaw offset at a safe hang --
        # with the payload under the body's yaw axis, the post-grasp
        # turn is a true yaw-in-place (an extended forward payload
        # sweeps a ~0.2 m arc that reads as pivoting around the stand)
        import numpy as _np
        _c = _np.where((_np.abs(ik._offsets[:, 1]) < 0.06)
                       & (ik._offsets[:, 2] < -0.12)
                       & (ik._offsets[:, 2] > -0.22))[0]
        if len(_c):
            _k = int(_c[_np.argmin(_np.abs(ik._offsets[_c, 1]))])
            self.q_tuck = ik._grid[_k].copy()
            self.off_tuck = ik._offsets[_k].copy()
        else:
            self.q_tuck, self.off_tuck = self.q_carry, self.off_carry

    @staticmethod
    def _forward_carry(ik):
        import numpy as _np
        cand = _np.where((ik._offsets[:, 2] > -0.13)
                         & (ik._offsets[:, 2] < -0.08))[0]
        k = int(cand[_np.argmin(ik._offsets[cand, 1])])  # forward = -y
        return ik._grid[k].copy(), ik._offsets[k].copy()

    @staticmethod
    def _mirrored(ik, drop):
        import numpy as _np
        q, _ = ik.solve_drop(drop)
        k = int(_np.argmin(_np.linalg.norm(ik._grid - (-_np.asarray(q)),
                                           axis=1)))
        return ik._grid[k].copy(), ik._offsets[k].copy()

    def reset(self, start_xyz, yaw=0.0):
        self.sp = np.array(start_xyz, dtype=float)
        self.grip = 1.0                       # open
        self.goal = self.sp.copy()
        self.goal_grip = 1.0
        self.arm = self.q_travel
        self.yaw = float(yaw)                 # slewed yaw setpoint
        self.goal_yaw = float(yaw)
        self.slow = False                     # final-approach creep flag

    def set_goal(self, xyz=None, grip=None, arm=None, yaw=None):
        if xyz is not None:
            self.goal = np.array(xyz, dtype=float)
        if grip is not None:
            self.goal_grip = float(grip)
        if arm is not None:
            self.arm = arm
        if yaw is not None:
            # wrap-aware (review 2026-08-23, "why is the drone spinning
            # while carrying"): remap the target to its numerically
            # nearest 2pi-equivalent of the CURRENT slewed yaw so the
            # ramp always turns the SHORT way. A beak-aligned grasp
            # ending near -pi followed by a bin turn near +pi walked
            # the ~2pi numeric gap as a full spin, still turning when
            # the 8 s wait timed out and the carry leg began.
            self.goal_yaw = self.yaw + float(
                (yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi)

    def tick(self):
        """Slew sp/grip/yaw one tick toward the goal; command the
        platform; return the action just commanded, as per-step deltas.

        The dyaw channel is live (user request 2026-08-19; it also matches
        the paper's 4-DoF x,y,z,yaw action space): the expert turns the
        nose toward where it is about to fly, so the scene camera actually
        watches the task. Commanded turns stay inside (-pi, pi); the PD's
        yaw error is wrap-aware regardless (pd_flight.py)."""
        # carry-rate guard covers ANY held object (workflow review: the
        # old < 0.5 threshold missed the penguin's close=0.67, so
        # penguin transits ran at 0.35 m/s -- 3x the documented rate)
        step = SP_STEP_CARRY if self.goal_grip < 0.95 else SP_STEP
        if self.slow:
            step = SP_STEP_FINAL
        # D41 (user review): VECTOR slew with distance-proportional
        # deceleration. The old per-axis clip flew L-infinity ramps --
        # the z-axis finished first, so every approach DIVED to target
        # altitude early and cruised low beside the stand ("down the
        # pole then up"); and the constant rate ended in a hard stop
        # the body glided past ("lunges forward then backs up"). The
        # setpoint now moves ALONG the straight 3D line to the goal
        # (true diagonals; z can never undershoot the target height on
        # a descent), at a rate that tapers with remaining distance
        # (P-controller arrival: rate = min(step, 0.18*dist) per tick,
        # floored so it always converges) -- fine control engages
        # automatically near the goal, no glide, no brake waypoints.
        err = self.goal - self.sp
        dist = float(np.linalg.norm(err))
        if dist > 1e-9:
            rate = min(step, max(0.5 * SP_STEP_FINAL, 0.18 * dist))
            d = err * (min(rate, dist) / dist)
        else:
            d = np.zeros(3)
        self.sp = self.sp + d
        dg = np.clip(self.goal_grip - self.grip, -GRIP_STEP, GRIP_STEP)
        self.grip = self.grip + dg
        ystep = YAW_STEP_CARRY if self.goal_grip < 0.95 else YAW_STEP
        dyaw = float(np.clip(self.goal_yaw - self.yaw, -ystep, ystep))
        self.yaw += dyaw
        self.ctrl.set_targets(self.sp, self.arm,
                              C.GRIPPER_OPEN * self.grip)
        self.ctrl.mppi.target_yaw = self.yaw
        return np.array([d[0], d[1], d[2], 0.0, 0.0, dyaw, self.grip],
                        dtype=np.float32)


# ---------------------------------------------------------------------------
# episode programs
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, seed):
        self.ctrl = C.SkyGripController(MODEL, C.MPPIParams(50, 16),
                                        flight="pd")
        self.model, self.data = self.ctrl.model, self.ctrl.data
        self.ik = C.ArmIK(self.model)
        self.expert = Expert(self.ctrl, self.ik)
        self.rng = np.random.default_rng(seed)
        # native-512 render, stored directly (D33); the old 2x-super-
        # sampled 256 pipeline is superseded -- at 512 the aliasing the
        # supersampling fought is below the train-time resize kernel
        self.rend = mujoco.Renderer(self.model, height=IMG, width=IMG)
        m = self.model
        self.bot = m.body("mustard_bottle").id
        self.badr = m.jnt_qposadr[m.body_jntadr[self.bot]]
        self.wbody = m.body("pick_weight").id
        self.wadr = m.jnt_qposadr[m.body_jntadr[self.wbody]]
        self.weld = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      "grasp_weld")
        self.gadr = m.joint("right_clamp").qposadr[0]
        self.gate = m.body("gate").id
        self.base_mass = float(m.body_mass[self.bot])
        self.base_inertia = m.body_inertia[self.bot].copy()
        self.sub = max(1, round((1.0 / FPS) / m.opt.timestep))
        # Task-object table: everything episode code needs, keyed by the
        # name used in the prompt. aim_z / close are the validated grasp
        # parameters per object (weight: stem pinch, seat ~7.5 mm; bottle:
        # cap pinch per D8, seat ~11 mm on the 23 mm cap). park is where
        # the object sits as background clutter when it is not the task.
        self.objs = {
            # narrow = pre-narrow grip fraction; gate_xy = alignment
            # gate radius. BOTH are clearance-driven per object: the
            # 16 mm stem leaves 4 mm/side under a 24 mm pre-narrow, but
            # the 23 mm cap leaves only 0.5 mm/side -- a pre-narrow at
            # the gate's 4.5 mm worst-case alignment TOUCHES the cap and
            # shoves the bottle (measured: -9.9 mm pre-close, object
            # knocked 0.14 m). The bottle pre-narrows to 27 mm and gates
            # at 2 mm so the pads can never meet the cap off-centre.
            # gate is ANISOTROPIC: gate_x guards the CLOSING axis (a pad
            # meeting the object off-centre is the graze/knock mechanism);
            # gate_y is the mouth-depth axis, where several mm just seat
            # the object deeper or shallower in the jaws -- measured
            # landings scatter +-6 mm in y while x stays sub-mm.
            # backoff = deliberate mouth-depth shortfall along the
            # approach axis (user 2026-08-27: "gripper pushed too far
            # into the object, jiggles it -- gentler"): the creep
            # stops this many m short of dead-centre, so the palm
            # never presses the object pre-close. The vertical stem
            # tolerates 8 mm; the plush head pinch is depth-sensitive
            # and gets 3 mm.
            "weight": dict(body=self.wbody, adr=self.wadr,
                           aim_z=WEIGHT_AIM_Z, close=0.45,
                           narrow=0.75, gate_x=0.0045, gate_y=0.0065,
                           ap_lo=5.0, ap_hi=12.0, stage=0.04,
                           backoff=0.008,
                           park=(-0.75, 1.25)),
            # plush penguin (2026-08-20, the paper's own manipuland,
            # replaces the mustard bottle on user request): 22 mm head
            # pinch, weight-family overhead grasp. aim = head centre.
            "plush penguin": dict(body=m.body("penguin").id,
                              adr=m.jnt_qposadr[m.body_jntadr[
                                  m.body("penguin").id]],
                              aim_z=0.083, close=0.67,
                              narrow=0.84, gate_x=0.0020,
                              gate_y=0.0065,
                              ap_lo=9.0, ap_hi=13.5, stage=0.04,
                              backoff=0.003,
                              park=(-0.55, 1.05)),
        }
        self.cur = self.objs["weight"]
        self.mustard_adr = self.badr   # bottle = scenery only now
        self.bin_id = m.body("bin").id
        self.bin2_id = m.body("bin2").id
        self.stand_id = m.body("stand").id
        self.stand2_id = m.body("stand2").id
        self.table_id = m.body("table").id
        self.bin_geoms = [m.geom(n).id for n in
                          ("bin_floor", "bin_wall_xlo", "bin_wall_xhi",
                           "bin_wall_ylo", "bin_wall_yhi")]
        self.bin2_geoms = [m.geom(n).id for n in
                           ("bin2_floor", "bin2_wall_xlo", "bin2_wall_xhi",
                            "bin2_wall_ylo", "bin2_wall_yhi")]
        self.bin_xy = np.array(BIN_XY)

    # -- low-level helpers ---------------------------------------------------
    def jaws(self):
        return C.grasp_site_pos(self.model, self.data)

    def weld_grasp(self, on):
        """Grasp weld (scene XML, doc L7): activated only from a MEASURED
        seated pinch, released on open. Encodes the hardware's demonstrated
        100 g carry at 0.2-0.5 N, which the contact solver cannot reproduce
        with 3-gram fingertips."""
        m, d = self.model, self.data
        if not on:
            d.eq_active[self.weld] = 0
            return
        b1 = m.body("gripper_assembly").id
        p1, q1 = d.xpos[b1], d.xquat[b1]
        p2, q2 = d.xpos[self.cur["body"]], d.xquat[self.cur["body"]]
        q1i = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        rel = np.zeros(3)
        mujoco.mju_rotVecQuat(rel, p2 - p1, q1i)
        rq = np.zeros(4)
        mujoco.mju_mulQuat(rq, q1i, q2)
        m.eq_data[self.weld][0:3] = 0.0
        m.eq_data[self.weld][3:6] = rel
        m.eq_data[self.weld][6:10] = rq
        m.eq_data[self.weld][10] = 1.0
        d.eq_active[self.weld] = 1

    def state(self):
        d = self.data
        return np.concatenate([
            d.qpos[0:7],
            [d.qpos[self.gadr] / 0.016],          # aperture fraction
            d.qpos[7:9],
        ]).astype(np.float32)

    def scene_state(self):
        m, d = self.model, self.data
        other = [o for k, o in self.objs.items() if o is not self.cur][0]
        b2 = m.body_pos[self.bin2_id]
        q2 = m.body_quat[self.bin2_id]
        yaw2 = 2 * np.arctan2(q2[3], q2[0])
        return np.concatenate([
            d.qpos[self.cur["adr"]:self.cur["adr"] + 7],
            d.qpos[other["adr"]:other["adr"] + 7],
            [self.bin_xy[0], self.bin_xy[1], self.bin_yaw],
            [b2[0], b2[1], yaw2],
        ]).astype(np.float32)

    def frame(self, task):
        f = {"observation.state": self.state(),
             "scene_state": self.scene_state(),
             "action": None,                      # filled by step()
             "task": task}
        for key, cam in CAMS.items():
            self.rend.update_scene(self.data, camera=cam)
            f[f"observation.images.{key}"] = self.rend.render().copy()
        return f

    def step(self, frames, task):
        """One 10 Hz tick: record obs, command expert, advance physics."""
        f = self.frame(task)
        f["action"] = self.expert.tick()
        for _ in range(self.sub):
            self.ctrl.step()
            mujoco.mj_step(self.model, self.data)
        frames.append(f)

    def run_until(self, frames, task, done, timeout_s, min_hold=3):
        """Tick until `done()` holds for `min_hold` consecutive ticks."""
        held = 0
        for _ in range(int(timeout_s * FPS)):
            self.step(frames, task)
            held = held + 1 if done() else 0
            if held >= min_hold:
                return True
        return False

    def settle_near(self, frames, task, tol=0.006, timeout_s=10.0):
        ok = self.run_until(
            frames, task,
            lambda: np.linalg.norm(self.data.qpos[0:3] - self.expert.goal) < tol,
            timeout_s)
        self.run_until(frames, task, lambda: False, timeout_s=0.6)  # hold:
        # measurements at the arrival tick are transients (measured diverging)
        return ok

    def hold_xy_until(self, frames, task, base_xy, z_target, done,
                      timeout_s):
        """Tick until done(), holding the BODY at base_xy while the
        z-setpoint tracks z_target. Pilot-style trim compensation: the
        loaded P-only PD parks the body its trim offset PAST the
        setpoint, and that offset changes the instant the payload welds
        on or releases -- aiming the sp short by the live estimate
        (body - sp) keeps the body on station through the transient
        instead of letting it swing nose-ward."""
        ex = self.expert
        for _ in range(int(timeout_s * FPS)):
            corr = self.data.qpos[0:2] - ex.sp[0:2]
            ex.set_goal(xyz=np.array([base_xy[0] - corr[0],
                                      base_xy[1] - corr[1],
                                      float(z_target)]))
            self.step(frames, task)
            if done():
                return True
        return False

    # -- scene randomisation -------------------------------------------------
    def reset_scene(self, task_xy, drone_xyz, gate_xy=None,
                    bottle_mass=BOTTLE_MASS, yaw=0.0, obj="weight"):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        k = bottle_mass / self.base_mass
        m.body_mass[self.bot] = self.base_mass * k
        m.body_inertia[self.bot] = self.base_inertia * k
        # the episode's task object goes to the task spot; the OTHER
        # candidate object rides its own stand, placed further down
        # once the bin position is known (its clearance constraints
        # need it)
        self.cur = self.objs[obj]
        for key, o in self.objs.items():
            if key == obj:
                d.qpos[o["adr"]:o["adr"] + 2] = task_xy
            # random yaw per episode (review 2026-08-20: rotated penguin
            # demos) -- the penguin faces a random direction; the grasp
            # is orientation-agnostic (spherical head pinch)
            t = self.rng.uniform(0, 2 * np.pi)
            d.qpos[o["adr"] + 3:o["adr"] + 7] = [np.cos(t / 2), 0, 0,
                                                 np.sin(t / 2)]
        # TABLE ERA (2026-08-26, supersedes the D40 pedestals): the
        # task object sits near the table's FRONT (+y, room-side)
        # edge -- 9 cm inside it, so the grasp hover keeps the front
        # legs off the surface -- and the table slides under it. The
        # top is at PLATE_TOP, so every validated grasp-altitude
        # constant carries over. The pedestals park underground.
        txy = np.asarray(task_xy, float)
        u_off = (float(self.rng.choice([-1, 1]))
                 * self.rng.uniform(0.10, 0.28))
        self._table_cx = float(np.clip(txy[0] + u_off, -0.55, 0.55))
        m.body_pos[self.table_id][0:2] = [self._table_cx,
                                          txy[1] - 0.21]
        m.body_pos[self.table_id][2] = MAT_TOP
        m.body_pos[self.stand_id][2] = -3.0
        m.body_pos[self.stand2_id][2] = -3.0
        d.qpos[self.cur["adr"] + 2] = PLATE_TOP + 0.005
        if obj == "plush penguin":
            # the task-penguin faces INTO the table (beak -y, +-0.7):
            # the behind-the-beak grasp then always approaches from
            # the room side, never over the table
            tb = self.rng.uniform(-0.7, 0.7)
            ap = self.cur["adr"]
            d.qpos[ap + 3:ap + 7] = [np.cos(tb / 2), 0, 0,
                                     np.sin(tb / 2)]
        # the mustard bottle is scenery now: park it out of the workspace
        d.qpos[self.mustard_adr:self.mustard_adr + 2] = (-0.95, 2.45)
        d.qpos[self.mustard_adr + 2] = 0.0
        d.qpos[self.mustard_adr + 3:self.mustard_adr + 7] = [1, 0, 0, 0]
        d.eq_active[self.weld] = 0
        m.eq_obj2id[self.weld] = self.cur["body"]   # weld follows the task
        # Anti-overfit scene variety (review 2026-08-19): the target box
        # moves and changes wood tone per episode, and a second
        # differently-coloured distractor box moves independently.
        self.bin_xy = np.array([self.rng.uniform(1.1, 1.7),
                                self.rng.uniform(1.2, 2.0)])
        m.body_pos[self.bin_id][0:2] = self.bin_xy
        m.body_pos[self.bin_id][2] = MAT_TOP
        # boxes also spawn ROTATED (review 2026-08-20); the placed check
        # tests the object inside the rotated box frame
        self.bin_yaw = float(self.rng.uniform(-np.pi, np.pi))
        m.body_quat[self.bin_id] = [np.cos(self.bin_yaw / 2), 0, 0,
                                    np.sin(self.bin_yaw / 2)]
        t2 = self.rng.uniform(-np.pi, np.pi)
        m.body_quat[self.bin2_id] = [np.cos(t2 / 2), 0, 0, np.sin(t2 / 2)]
        m.body_pos[self.bin2_id][0:2] = [self.rng.uniform(-1.3, -0.7),
                                         self.rng.uniform(1.8, 2.7)]
        m.body_pos[self.bin2_id][2] = MAT_TOP
        r0 = self.rng.uniform(0.5, 1.0)
        tint = [r0, r0 * self.rng.uniform(0.6, 0.85),
                r0 * self.rng.uniform(0.35, 0.6), 1.0]
        for g in self.bin_geoms:
            m.geom_rgba[g] = tint
        # distractor stays LIGHT and COOL (pale grey/blue/green): a dark
        # random tint was hard to tell from a dark-wood target box
        # (review 2026-08-19)
        b0 = self.rng.uniform(0.65, 0.95)
        tint2 = [b0 * self.rng.uniform(0.60, 0.90),
                 b0 * self.rng.uniform(0.85, 1.05),
                 b0 * self.rng.uniform(0.95, 1.15), 1.0]
        for g in self.bin2_geoms:
            m.geom_rgba[g] = np.clip(tint2, 0.05, 1.0)
        # the DISTRACTOR shares the SAME table edge (user 2026-08-26:
        # both candidates on one ordinary surface, so language -- not
        # furniture -- selects the target), >=0.40 m along the edge so
        # the grasp hover keeps its clearance
        cx = self._table_cx
        ra = (cx + 0.36) - txy[0]
        la = txy[0] - (cx - 0.36)
        sside, avail = (1.0, ra) if ra >= la else (-1.0, la)
        dsep = self.rng.uniform(0.45, max(0.46, min(0.62, avail)))
        for key, o in self.objs.items():
            if key != obj:
                d.qpos[o["adr"]:o["adr"] + 2] = [txy[0] + sside * dsep,
                                                 txy[1]]
                d.qpos[o["adr"] + 2] = PLATE_TOP + 0.005
        if gate_xy is not None:
            m.body_pos[self.gate][0:2] = gate_xy
            m.body_pos[self.gate][2] = MAT_TOP
        else:
            # No gate in pick-and-place episodes (review 2026-08-19):
            # park it 3 m underground; nav episodes restore it.
            m.body_pos[self.gate][2] = -3.0
        d.qpos[0:3] = drone_xyz
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:9] = self.expert.q_travel
        mujoco.mj_forward(m, d)
        self.ctrl.reset_after_randomisation()
        self.ctrl._q_cmd = np.array(self.expert.q_travel, float)
        self.ctrl.mppi.target_yaw = float(yaw)
        self.expert.reset(drone_xyz, yaw=yaw)

    def object_settled(self):
        """True when the task object is at rest -- and, for the bottle,
        actually STANDING. Guards the upright-path weld: the weld once
        caught a bottle MID-TIP at a transiently-aligned instant and
        carried it frozen at a tilt, 'picked up as if it was upright'
        (review, episode 4). A tipping bottle must read as a miss so the
        retry loop reorients and runs the proper fallen recovery."""
        vadr = self.model.jnt_dofadr[
            self.model.body_jntadr[self.cur["body"]]]
        return float(np.linalg.norm(
            self.data.qvel[vadr:vadr + 6])) < 0.10

    def side_grasp(self, frames, task, ex, aim):
        """Horizontal mouth-entry grasp of the upright bottle cap.
        Returns (welded, gate_ok, pre_close, entry_ok)."""
        so = aim + SIDE_STANDOFF * SIDE_ENTRY
        ex.set_goal(xyz=so - OFF_SIDE, arm=Q_SIDE)
        self.settle_near(frames, task, tol=0.03)
        for _ in range(3):
            resid = self.jaws() - so
            if abs(resid[0]) < 0.002 and abs(resid[2]) < 0.004:
                break
            ex.set_goal(xyz=ex.goal - np.array([resid[0], 0.0, resid[2]]))
            self.settle_near(frames, task, tol=0.012)
        # creep entry, live-gated on the closing axis only (the entry
        # path itself starts 34 mm off the endpoint by design)
        ex.slow = True
        ex.set_goal(xyz=aim - OFF_SIDE)
        entry_ok = True
        for _ in range(int(10.0 * FPS)):
            self.step(frames, task)
            dv = self.jaws() - aim
            if abs(dv[0]) > 0.012:
                entry_ok = False
                break
            if np.linalg.norm(dv) < 0.006:
                break
        ex.slow = False
        if not entry_ok:
            return False, False, self.jaws() - aim, False
        narrow = self.cur["narrow"]
        ex.set_goal(grip=narrow)
        self.run_until(frames, task,
                       lambda: abs(ex.grip - narrow) < 0.01,
                       timeout_s=0.8, min_hold=1)
        pre_close = self.jaws() - aim
        # upright-only object check: with the cap inside the open mouth,
        # pad-contact solver jitter keeps any strict at-rest threshold
        # unreachable (measured); the mid-tip cheat still shows as
        # R22 << 1 and is caught
        adr = self.cur["adr"]
        q = self.data.qpos[adr + 3:adr + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        upright = R.reshape(3, 3)[2, 2] > 0.95
        gate_ok = bool(abs(pre_close[0]) < 0.004
                       and np.linalg.norm(pre_close) < 0.008 and upright)
        welded = False
        if gate_ok:
            self.weld_grasp(True)
            i_pre = self.ctrl.mppi._i_pos.copy()
            close = self.cur["close"]
            ex.set_goal(grip=close)
            self.run_until(frames, task,
                           lambda: abs(ex.grip - close) < 0.01,
                           timeout_s=2.0)
            ap = float(self.data.qpos[self.gadr]) * 1000
            if self.cur["ap_lo"] < ap < self.cur["ap_hi"]:
                welded = True
                self.weld_grasp(True)           # datum re-capture
                self.ctrl.mppi._i_pos[:] = i_pre
            else:
                self.weld_grasp(False)
        return welded, gate_ok, pre_close, True

    def live_target(self, fallen_ok=True):
        """Grasp aim from the object's LIVE pose (not the spawn pose):
        a nudged object is re-aimed at wherever it actually is. Both
        task objects (weight, red block) are tip- and roll-stable, so
        the top-down aim is always valid; the bottle-era fallen-recovery
        machinery retired with the bottle (D30)."""
        adr = self.cur["adr"]
        pos = self.data.qpos[adr:adr + 3]
        aim = np.array([pos[0], pos[1],
                        float(pos[2]) + self.cur["aim_z"]])
        # HEADING-AWARE grasp for the penguin (2026-08-21): its beak is
        # solid and protrudes at head height, so the drone yaws to match
        # the penguin's facing -- the beak points out through the open
        # jaw mouth and the pads land on the SIDES of the head, never
        # the beak.
        if self.cur is self.objs["plush penguin"]:
            q = self.data.qpos[adr + 3:adr + 7]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            beak = R.reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
            yaw = float(np.arctan2(beak[0], -beak[1]))
            return aim, yaw
        return aim, None

    # -- the three programs --------------------------------------------------
    def manip_episode(self, obj="weight", fallen_start=False,
                      offset_hover=False):
        rng = self.rng
        # object spawn varies widely too (review 2026-08-19)
        bxy = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.20, 1.00)])
        # D42 history: a spawn ABOVE grasp altitude near the stand made
        # the straight 3D leg plunge beside the pole ("flying to the
        # base of the pole"); level spawns fixed it, and the up-then-
        # across takeoff below supersedes both -- altitude is gained
        # vertically at the spawn, never shed en route.
        alt0 = (PLATE_TOP + 0.005 + self.objs[obj]["aim_z"]
                - float(self.expert.off_carry[2]))
        # corrective flavour: the STAND relocates to the workspace edge
        # -- draw that position FIRST so the spawn-geometry rejection
        # below validates against where the stand will actually be
        # (workflow review: the old post-hoc teleport could leave the
        # drone 0.25 m from the pole with altitude to dump)
        exy = None
        if fallen_start:
            # 0.88 cap keeps the edge-spawned object inside the
            # table's on-top zone once the table centre clamps at 0.55
            edge_x = float(rng.choice([-1, 1])) * rng.uniform(0.72, 0.88)
            exy = np.array([edge_x, rng.uniform(0.1, 1.1)])
        anchor = exy if exy is not None else bxy
        # UP-THEN-ACROSS (user 2026-08-26 third pass: "go up and then
        # across instead of across then up"): the drone spawns LOW and
        # the episode opens with a vertical climb to grasp altitude at
        # the spawn spot; every transit after that is dead level. The
        # low spawn also restores the z-variety the level-spawn fix
        # removed. Spawn keeps a real approach leg (>= 0.70 m out).
        # spawn near the DECK (user 2026-08-26: "more height on the
        # initial rise"): body 0.21-0.30 puts the leg tips just above
        # the mat, so the opening climb to grasp altitude is a proper
        # 25-35 cm takeoff; it still tops out exactly at alt0, so the
        # cruise that follows stays dead level
        for _ in range(300):
            start = np.array([rng.uniform(-0.5, 0.5),
                              rng.uniform(0.9, 1.6),
                              rng.uniform(0.21, 0.30)])
            if (float(np.linalg.norm(start[0:2] - anchor)) >= 0.70
                    and start[1] >= anchor[1] + 0.35):
                # room-side of the table's front edge (at anchor_y +
                # 0.09), so the deck takeoff never sits over the top
                break
        # corrective passes its edge position straight in as the task
        # spot -- a post-reset teleport would invalidate the distractor
        # stand's clearance constraints
        self.reset_scene(exy if fallen_start else bxy, start, obj=obj)
        ex, frames = self.expert, []
        task = PROMPT_MANIP.format(obj=obj)
        ok = True
        self.run_until(frames, task, lambda: False, timeout_s=1.0)
        # (0) UP first, THEN across: vertical climb at the spawn spot
        # to grasp altitude, stabilise, and only then begin the level
        # transit toward the object
        ex.set_goal(xyz=np.array([start[0], start[1], alt0]))
        self.run_until(frames, task,
                       lambda: (abs(self.data.qpos[2] - alt0) < 0.04
                                and abs(self.data.qvel[2]) < 0.05),
                       timeout_s=8.0)
        self.run_until(frames, task, lambda: False, timeout_s=0.5)

        # ------ D40 LATERAL ORBIT GRASP (replaces the overhead grasp;
        # probe-validated 6/6 across both objects, 2026-08-26). The
        # object sits on the stand at flight altitude and stays in the
        # wrist + forward cameras from approach to pinch -- the fix for
        # the D37 blind-descent ceiling. Flight profile per Hana's
        # review: turn in place, braked slow approach to the ring, arm
        # extension at dead hover, orbit to the beak bearing (penguin;
        # the weight is symmetric and is grasped from the arrival
        # bearing), single slow creep, anisotropic close gate, weld.
        adr = self.cur["adr"]
        welded, gate_ok, grasp_try = False, False, 0
        pre_close = np.zeros(3)
        ap = 0.0
        u = np.array([0.0, 1.0])
        for grasp_try in range(3):
            if grasp_try > 0:
                # retry = LEVEL back-out to the ring for another pass
                # (user 2026-08-26: the old climb-0.2-then-redescend
                # read as a down-up bounce beside the pole -- and it
                # dodged the approach gates, which only watched up to
                # the FIRST close). The reopened mouth slides straight
                # back off the object along the approach axis, so
                # altitude never changes.
                ex.slow = True
                ex.set_goal(xyz=np.array([pre_pt[0], pre_pt[1],
                                          float(ex.goal[2])]),
                            grip=1.0)
                self.settle_near(frames, task, tol=0.05, timeout_s=12.0)
                ex.slow = False
            aim, yaw_g = self.live_target()
            if float(aim[2]) < PLATE_TOP - 0.05:
                # object knocked off the stand (workflow review): the
                # lateral corridor at fallen-object altitude would fly
                # below the plate into the shaft -- fail cleanly and
                # let the banking filter discard the episode
                break
            alt = float(aim[2] - ex.off_carry[2])
            pxy = aim[0:2].copy()
            # grasp bearing: BEHIND the penguin (live heading); the
            # symmetric weight is approached from the arrival bearing
            if yaw_g is not None:
                q4 = self.data.qpos[adr + 3:adr + 7]
                R9 = np.zeros(9)
                mujoco.mju_quat2Mat(R9, q4)
                beak = R9.reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
                u = beak[0:2] / max(1e-6,
                                    float(np.linalg.norm(beak[0:2])))
            else:
                here0 = self.data.qpos[0:2]
                u = pxy - here0
                u = u / max(1e-6, float(np.linalg.norm(u)))
                # table era: clamp the grasp bearing to <=45 deg off
                # the front-edge normal -- a fully side-on approach
                # would park the grasp hover too near the distractor
                # sharing the edge
                if -u[1] < 0.707:
                    sx = 1.0 if u[0] >= 0 else -1.0
                    u = np.array([sx * 0.707, -0.707])
            pre_pt = np.array([pxy[0] - RING_R * u[0],
                               pxy[1] - RING_R * u[1], alt])
            # routing (workflow review, replaces the dogleg): when the
            # beak-bearing ring point sits more than ~50 deg around the
            # stand, fly the D40 ORBIT for real -- direct level leg to
            # the NEAREST ring point, then walk the ring in <=45 deg
            # hops at creep speed with the nose held on the object
            # (user 2026-08-21: "its good the penguin is always in
            # camera"). Chords between ring points <=50 deg apart stay
            # >=0.41 m from the pole, so no leg ever cuts the ring;
            # the old dogleg's 90-deg bypass chord passed 0.32 m out
            # at full travel speed.
            here2 = np.array(self.data.qpos[0:2], float)
            th_me = float(np.arctan2(here2[1] - pxy[1],
                                     here2[0] - pxy[0]))
            th_t = float(np.arctan2(pre_pt[1] - pxy[1],
                                    pre_pt[0] - pxy[0]))
            gap = float((th_t - th_me + np.pi) % (2 * np.pi) - np.pi)
            if abs(gap) > 0.9:
                near_pt = np.array([pxy[0] + RING_R * np.cos(th_me),
                                    pxy[1] + RING_R * np.sin(th_me),
                                    alt])
                to0 = near_pt[0:2] - here2
                if float(np.linalg.norm(to0)) > 0.10:
                    ex.set_goal(yaw=float(np.arctan2(to0[0], -to0[1])))
                    self.run_until(frames, task,
                                   lambda: abs(ex.yaw - ex.goal_yaw)
                                   < 0.03, timeout_s=10.0)
                ex.set_goal(xyz=near_pt,
                            arm=(ex.q_travel if grasp_try == 0
                                 else ex.q_carry))
                self.settle_near(frames, task, tol=0.06, timeout_s=12.0)
                n_hop = int(np.ceil(abs(gap) / 0.785))
                ex.slow = True
                for k in range(1, n_hop + 1):
                    th_k = th_me + gap * k / n_hop
                    wp = np.array([pxy[0] + RING_R * np.cos(th_k),
                                   pxy[1] + RING_R * np.sin(th_k),
                                   alt])
                    tv = pxy - self.data.qpos[0:2]
                    ex.set_goal(xyz=wp, yaw=float(np.arctan2(tv[0],
                                                             -tv[1])))
                    self.settle_near(frames, task, tol=0.07,
                                     timeout_s=10.0)
                ex.slow = False
            # (1) turn IN PLACE toward the pre-grasp point, then ONE
            # straight leg there (review 2026-08-26: "move directly" --
            # a single diagonal that reaches grasp altitude only on
            # arrival; no low cruise beside the stand, no orbit)
            here = self.data.qpos[0:2]
            to = pre_pt[0:2] - here
            if float(np.linalg.norm(to)) > 0.10:
                ex.set_goal(yaw=float(np.arctan2(to[0], -to[1])))
                self.run_until(frames, task,
                               lambda: abs(ex.yaw - ex.goal_yaw)
                               < 0.03, timeout_s=10.0)
            ex.set_goal(xyz=pre_pt, arm=(ex.q_travel if grasp_try == 0
                                         else ex.q_carry))
            ok &= self.settle_near(frames, task, tol=0.04)
            self.run_until(frames, task,
                           lambda: float(np.linalg.norm(
                               self.data.qvel[0:2])) < 0.03,
                           timeout_s=6.0)
            # (2) face the object, then verify a STABLE dead hover
            # before the arm moves (user 2026-08-26: "ensure robot is
            # stable hover then reach out arm to grasp") -- stillness
            # in translation AND rotation, HELD for half a second, not
            # just a yaw-setpoint arrival: the extension shifts the
            # CoM, and swinging the arm out of a still-moving hover
            # re-excites exactly the sway the creep then chases.
            to = pxy - self.data.qpos[0:2]
            ex.set_goal(yaw=float(np.arctan2(to[0], -to[1])))
            self.run_until(frames, task,
                           lambda: abs(ex.yaw - ex.goal_yaw) < 0.02,
                           timeout_s=8.0)
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(
                                        self.data.qvel[0:3])) < 0.025
                                    and float(np.linalg.norm(
                                        self.data.qvel[3:6])) < 0.10),
                           timeout_s=8.0, min_hold=5)
            if grasp_try == 0:
                ex.set_goal(arm=ex.q_carry)
                self.run_until(frames, task,
                               lambda: float(np.linalg.norm(
                                   self.data.qpos[7:9] - ex.q_carry))
                               < 0.06, timeout_s=14.0)
                # re-settle after the swing -- SHORTENED (user
                # 2026-08-27: "after the arm straightens out there's a
                # pause"): a looser exit (0.04/0.12, held 0.3 s, cap
                # 4 s) releases the creep ~1-2 s sooner; the strict
                # pre-close stillness check downstream remains the
                # hard guarantee on grasp quality
                self.run_until(frames, task,
                               lambda: (float(np.linalg.norm(
                                            self.data.qvel[0:3]))
                                        < 0.04
                                        and float(np.linalg.norm(
                                            self.data.qvel[3:6]))
                                        < 0.12),
                               timeout_s=4.0, min_hold=3)
            # composure beat before the creep (0.7 s, user 2026-08-27:
            # settled on 0.7 after trying 1.0/0.4/0.1 -- a readable
            # pause for grasp quality without the old full-second wait)
            self.run_until(frames, task, lambda: False, timeout_s=0.7)
            cyw, syw = np.cos(ex.yaw), np.sin(ex.yaw)
            offw = np.array(
                [cyw * ex.off_carry[0] - syw * ex.off_carry[1],
                 syw * ex.off_carry[0] + cyw * ex.off_carry[1],
                 ex.off_carry[2]])
            axp = np.array([u[1], -u[0]])
            # creep target sits `backoff` short of dead-centre along
            # the approach axis (user 2026-08-27: gentler -- the palm
            # was pressing the object pre-close and jiggling it)
            bk3 = self.cur.get("backoff", 0.0) * np.array(
                [u[0], u[1], 0.0])
            ex.slow = True
            reached = False
            for _ in range(4):
                aim, _ = self.live_target()
                if float(aim[2]) < PLATE_TOP - 0.05:
                    break            # fell mid-creep: abort this pass
                ex.set_goal(xyz=aim - bk3 - offw)
                if self.run_until(frames, task, lambda: (
                        float(np.linalg.norm(
                            (self.jaws() + bk3
                             - self.live_target()[0])[0:2])) < 0.005
                        and abs(float(
                            (self.jaws()
                             - self.live_target()[0])[2])) < 0.012),
                        timeout_s=14.0):
                    reached = True
                    break
            ex.slow = False
            if not reached:
                continue
            # close only from verified stillness (user spec: "grasp at
            # steady hover") -- the creep gate is positional, and any
            # residual drift stacks onto the weld transient (measured
            # 0.088 m/s on the corrective's side-on approach)
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(
                                        self.data.qvel[0:3])) < 0.03
                                    and float(np.linalg.norm(
                                        self.data.qvel[3:6])) < 0.10),
                           timeout_s=3.0, min_hold=3)
            close = self.cur["close"]
            ex.set_goal(grip=close)
            self.run_until(frames, task,
                           lambda: abs(ex.grip - close) < 0.01,
                           timeout_s=1.5, min_hold=1)
            self.run_until(frames, task, lambda: False, timeout_s=0.4)
            aim, _ = self.live_target()
            pre_close = self.jaws() - aim
            gxv = float(pre_close[0] * axp[0] + pre_close[1] * axp[1])
            gyv = float(pre_close[0] * u[0] + pre_close[1] * u[1])
            ap = float(self.data.qpos[self.gadr])
            # in-mouth gate is UPRIGHT-ONLY (D29: pad contact keeps the
            # solver's object velocity jittering; an at-rest check
            # refuses every honest pinch)
            R9c = np.zeros(9)
            mujoco.mju_quat2Mat(R9c,
                                self.data.qpos[adr + 3:adr + 7])
            gate_ok = (abs(gxv) < 0.006 and abs(gyv) < 0.030
                       and abs(float(pre_close[2])) < 0.015
                       and float(R9c.reshape(3, 3)[2, 2]) > 0.90)
            if gate_ok and self.cur["ap_lo"] < ap * 1000 < self.cur["ap_hi"]:
                welded = True
                self.weld_grasp(True)
                break
            ex.set_goal(grip=1.0)
            self.run_until(frames, task, lambda: ex.grip > 0.99,
                           timeout_s=1.5, min_hold=1)
        self._diag = {"ap_close_mm": round(ap * 1000, 1),
                      "welded": welded, "retried": grasp_try > 0}
        if not welded:
            # failed grasp: END the episode instead of flying the whole
            # delivery choreography with empty jaws (workflow review:
            # the fall-through pantomimed a 30-40 s phantom drop-off,
            # then chased an impossible payload residual past the box).
            # The banking filter discards the episode either way.
            self.run_until(frames, task, lambda: False, timeout_s=0.5)
            b = self.data.qpos[self.cur["adr"]:self.cur["adr"] + 3]
            return frames, {"task": task, "kind": "manip", "pick": False,
                            "place": False, "success": False,
                            "gate": bool(gate_ok),
                            "pre_close_mm": [round(float(v) * 1000, 1)
                                             for v in pre_close],
                            "end_obj": [round(float(v), 3) for v in b],
                            **getattr(self, "_diag", {})}
        # ---- SIMPLE CARRY (user spec 2026-08-26): grasp -> hover ->
        # yaw on the spot -> fly to the box -> drop. The arm stays
        # EXTENDED from the grasp pause to box arrival: zero arm motion
        # IN FLIGHT -- the payload rides rigidly on the nose through
        # the turn and cruise. The one arm move after grasp is the
        # release tuck at the stationary hover over the box (see the
        # KICKLESS RELEASE block for why physics demands it).
        # Station-holding starts AT THE WELD TICK (user 2026-08-26:
        # "grasp then hover, not grasp and swing forward"): the payload
        # shifts the equilibrium nose-ward the instant it welds on, and
        # the old uncompensated dwell + lift let the body swing before
        # the climb's correction engaged. hold_xy_until counters the
        # trim from tick one -- dwell, lift, climb and hover all pin
        # the body over the grasp spot.
        grasp_xy = np.array(self.data.qpos[0:2], float)
        at = np.array(ex.goal, float)
        self._lift_z0 = float(self.data.qpos[adr + 2])
        # (a0) grasp-confirmation dwell (as on the real platform)
        self.hold_xy_until(frames, task, grasp_xy, float(at[2]),
                           lambda: False, 0.3)
        # (a) straight ascent to hover altitude
        self.hold_xy_until(frames, task, grasp_xy, float(at[2]) + 0.12,
                           lambda: float(
                               self.data.qpos[self.cur["adr"] + 2])
                           > self._lift_z0 + 0.06, 4.0)
        # The at-hover check is altitude-threshold + near-zero vertical
        # speed, not a position tolerance -- the loaded droop deadlocks
        # tight settles.
        hover_z = float(at[2]) + 0.28
        self.hold_xy_until(frames, task, grasp_xy, float(at[2]) + 0.45,
                           lambda: (self.data.qpos[2] > hover_z
                                    and abs(self.data.qvel[2]) < 0.05),
                           12.0)
        self.hold_xy_until(frames, task, grasp_xy, float(at[2]) + 0.45,
                           lambda: False, 1.5)   # hover, still held
        # (b) yaw IN PLACE toward the box. set_goal is wrap-aware, so
        # the turn always goes the short way about the body's own yaw
        # axis; the payload turns with the nose like a carried load.
        # Under load the PD trims to a steady BODY offset from its
        # setpoint in the nose direction (~0.28 m at 100 g; the P-only
        # trim needs that position error to hold the tilted hover).
        # Left alone, the equilibrium offset ROTATES with the heading,
        # so the body walks a wide arc during the turn (measured
        # 0.46 m with the weight). Compensate like a pilot: hold the
        # BODY still by counter-rotating the setpoint,
        #     sp(t) = body0 - R(yaw(t)) @ lead_body,
        # which starts exactly at the settled sp (no jump) and keeps
        # the trim force pointing at the same world spot all turn.
        # nose is the body -y axis: at yaw 0 it points along world -y,
        # so facing a world direction (tx, ty) needs yaw = atan2(tx, -ty)
        # the lead estimate is only valid at verified stillness
        # (workflow review: a lead sampled mid-transient mis-aims the
        # whole counter-rotation and the turn walks a real arc)
        self.run_until(frames, task,
                       lambda: (float(np.linalg.norm(
                                    self.data.qvel[0:3])) < 0.03
                                and float(np.linalg.norm(
                                    self.data.qvel[3:6])) < 0.10),
                       timeout_s=8.0, min_hold=5)
        here3 = np.array(self.data.qpos[0:3], float)
        lead_w = here3[0:2] - ex.sp[0:2]      # settled loaded trim
        cy0, sy0 = np.cos(ex.yaw), np.sin(ex.yaw)
        lead_b = np.array([cy0 * lead_w[0] + sy0 * lead_w[1],
                           -sy0 * lead_w[0] + cy0 * lead_w[1]])
        yaw_des = float(np.arctan2(self.bin_xy[0] - here3[0],
                                   -(self.bin_xy[1] - here3[1])))
        ex.set_goal(yaw=yaw_des)
        # near-pi turns need ~16 s at the loaded yaw rate
        for _ in range(int(22.0 * FPS)):
            cyt, syt = np.cos(ex.yaw), np.sin(ex.yaw)
            lw = np.array([cyt * lead_b[0] - syt * lead_b[1],
                           syt * lead_b[0] + cyt * lead_b[1]])
            ex.set_goal(xyz=np.array([here3[0] - lw[0],
                                      here3[1] - lw[1],
                                      float(ex.goal[2])]))
            self.step(frames, task)
            if abs(ex.yaw - ex.goal_yaw) < 0.02:
                break
        # settle to verified stillness before measuring hang and lead
        # for the bin leg (workflow review: the old fixed 0.8 s dwell
        # sampled them mid-relaxation, and the error lands 1:1 on
        # where the payload parks at the box)
        self.run_until(frames, task,
                       lambda: (float(np.linalg.norm(
                                    self.data.qvel[0:3])) < 0.03
                                and float(np.linalg.norm(
                                    self.data.qvel[3:6])) < 0.10),
                       timeout_s=8.0, min_hold=5)

        held = float(self.data.qpos[self.cur["adr"] + 2]) > 0.15
        # (c) ONE straight cruise leg that parks the PAYLOAD over the
        # bin. The nose faces the box, so the hanging payload leads the
        # body by the live-measured offset -- aim the payload, not the
        # body; the carry-rate slew keeps the whole leg at the gentle
        # 0.12 m/s. Arrival is SP-convergence + stillness (under load
        # the PD trims a steady offset from its waypoint, so a
        # body-vs-goal tolerance deadlocks); the settled-residual loop
        # below then trims the payload dead over the bin centre with
        # invisible cm-scale nudges before the release.
        hang = (self.data.qpos[self.cur["adr"]:self.cur["adr"] + 2]
                - self.data.qpos[0:2])
        # LEAD compensation, reinstated (user 2026-08-26: "why did it
        # overshoot the box"): under load the body parks its steady
        # trim offset PAST the setpoint toward the nose -- i.e. toward
        # the box -- so an uncompensated goal overshoots by the trim
        # (~0.1-0.28 m) and gets walked back. Aim the setpoint SHORT
        # by the trim measured at this settled loaded hover, and the
        # body's natural parking spot puts the payload dead on the bin.
        lead = np.array(self.data.qpos[0:2] - ex.sp[0:2], float)
        # cruise LEVEL at the hover altitude (workflow review: the old
        # hard-coded 0.75 goal shed 0.25-0.31 m of the just-gained
        # climb along the leg -- a long shallow dive at the box); the
        # spec is level leg, arrive over the bin, THEN descend
        ex.set_goal(xyz=np.array([*(self.bin_xy - hang - lead),
                                  float(ex.goal[2])]))
        self.run_until(frames, task,
                       lambda: (float(np.linalg.norm(ex.sp - ex.goal))
                                < 0.01
                                and float(np.linalg.norm(
                                    self.data.qvel[0:2])) < 0.04),
                       timeout_s=28.0)
        ex.slow = True
        arrived_bin = False
        for _ in range(3):
            w_xy = self.data.qpos[self.cur["adr"]:self.cur["adr"] + 2]
            resid = np.array([w_xy[0] - self.bin_xy[0],
                              w_xy[1] - self.bin_xy[1], 0.0])
            if np.linalg.norm(resid[:2]) < 0.08:
                arrived_bin = True
                break
            ex.set_goal(xyz=ex.goal - resid)
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(ex.sp - ex.goal))
                                    < 0.01
                                    and float(np.linalg.norm(
                                        self.data.qvel[0:2])) < 0.04),
                           timeout_s=6.0)
        if arrived_bin:
            # KICKLESS RELEASE (tau scan 2026-08-26: the weld-off kick
            # is 0.51 m/s under EVERY setpoint schedule -- it is the
            # 1 N load-step acting on the 0.2 m nose lever, so no
            # control trick removes it; the tuck-era releases were
            # clean because the lever was zero). Remove the LEVER:
            # tuck the payload under the body at a station-held
            # stationary hover over the box -- the one arm move the
            # hover-only rule always allowed -- re-centre it, descend
            # into the box mouth, and let go from a few cm up. The
            # load step becomes pure-vertical (a gentle bob), and the
            # drop is ~3-5 cm instead of 22.
            hxy = np.array(self.data.qpos[0:2], float)
            ex.set_goal(arm=ex.q_tuck)
            self.hold_xy_until(frames, task, hxy, float(ex.sp[2]),
                               lambda: float(np.linalg.norm(
                                   self.data.qpos[7:9]
                                   - np.asarray(ex.q_tuck))) < 0.08,
                               14.0)
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(
                                        self.data.qvel[0:3])) < 0.03
                                    and float(np.linalg.norm(
                                        self.data.qvel[3:6])) < 0.10),
                           timeout_s=6.0, min_hold=5)
            # re-centre the now-underslung payload on the box
            for _ in range(3):
                w_xy = self.data.qpos[self.cur["adr"]:
                                      self.cur["adr"] + 2]
                resid = np.array([w_xy[0] - self.bin_xy[0],
                                  w_xy[1] - self.bin_xy[1], 0.0])
                if np.linalg.norm(resid[:2]) < 0.05:
                    break
                ex.set_goal(xyz=ex.goal - resid)
                self.run_until(frames, task,
                               lambda: (float(np.linalg.norm(
                                            ex.sp - ex.goal)) < 0.01
                                        and float(np.linalg.norm(
                                            self.data.qvel[0:2]))
                                        < 0.04),
                               timeout_s=6.0)
            # descend until the payload hangs a few cm above the box
            # floor (body ~0.31: rotors 11 cm above the rim, leg tips
            # inside the 42 cm box mouth with 10 cm lateral clearance)
            hang_z = float(self.data.qpos[2]
                           - self.data.qpos[self.cur["adr"] + 2])
            ex.set_goal(xyz=np.array([float(ex.goal[0]),
                                      float(ex.goal[1]),
                                      MAT_TOP + 0.09 + hang_z]))
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(
                                        ex.sp - ex.goal)) < 0.01
                                    and abs(float(
                                        self.data.qvel[2])) < 0.05),
                           timeout_s=12.0)
            ex.set_goal(grip=1.0)
            self.run_until(frames, task, lambda: ex.grip > 0.99,
                           timeout_s=1.0, min_hold=1)
            rel_xy = np.array(self.data.qpos[0:2], float)
            self.weld_grasp(False)
            self.hold_xy_until(frames, task, rel_xy, float(ex.sp[2]),
                               lambda: False, 0.8)
            # straight GENTLE ascent back out of the box mouth (user
            # 2026-08-26: the post-place phase turned jerky -- every
            # move after the set-down stays at creep/drift pace)
            ex.slow = True
            self.hold_xy_until(frames, task, rel_xy, 0.9,
                               lambda: self.data.qpos[2] > 0.72, 14.0)
        # retract at a STATIONARY, station-held hover, then a SLOW
        # drift back (user 2026-08-26: the post-place full-speed dart
        # to the retreat point was the "suddenly jerky" ending -- the
        # whole wind-down now stays at drift pace)
        hold_xy = np.array(self.data.qpos[0:2], float)
        hold_z = float(ex.sp[2])
        ex.set_goal(arm=ex.q_travel)
        self.hold_xy_until(frames, task, hold_xy, hold_z,
                           lambda: float(np.linalg.norm(
                               self.data.qpos[7:9]
                               - np.asarray(ex.q_travel))) < 0.08,
                           10.0)
        # END at a PINNED hover (user 2026-08-27: "after it has placed
        # it, it remains at hover and doesn't drift") -- the old slow
        # retreat leg read as drifting, so it is gone. Wait out the
        # arm-fold's CoM transient to verified stillness FIRST, then
        # pin: the hold must anchor a settled body, not a moving one.
        self.run_until(frames, task,
                       lambda: (float(np.linalg.norm(
                                    self.data.qvel[0:3])) < 0.025
                                and float(np.linalg.norm(
                                    self.data.qvel[3:6])) < 0.10),
                       timeout_s=6.0, min_hold=5)
        ex.slow = False
        end_xy = np.array(self.data.qpos[0:2], float)
        self.hold_xy_until(frames, task, end_xy, float(ex.sp[2]),
                           lambda: False, 1.5)

        b = self.data.qpos[self.cur["adr"]:self.cur["adr"] + 3]
        # placed check in the ROTATED box frame
        dx, dy = b[0] - self.bin_xy[0], b[1] - self.bin_xy[1]
        cb, sb = np.cos(-self.bin_yaw), np.sin(-self.bin_yaw)
        bx, by = cb * dx - sb * dy, sb * dx + cb * dy
        placed = (abs(bx) < 0.21 and abs(by) < 0.21
                  and b[2] < MAT_TOP + 0.20)
        # Success = PLACED: an object cannot arrive at rest inside the bin
        # without having been picked and carried there. The mid-episode
        # `held` sample is diagnostic only -- it was measured misreading a
        # weld-carried lift and discarding a physically perfect episode.
        return frames, {"task": task, "kind": "manip", "pick": held,
                        "place": bool(placed),
                        "success": bool(placed),
                        "gate": bool(gate_ok),
                        "pre_close_mm": [round(float(v) * 1000, 1)
                                         for v in pre_close],
                        "end_obj": [round(float(v), 3) for v in b],
                        **getattr(self, "_diag", {})}

    def nav_episode(self, corrective=False, obj="weight"):
        rng = self.rng
        gate_xy = np.array(GATE_LEFT if rng.random() < 0.5 else GATE_RIGHT)
        bxy = np.array([rng.uniform(-0.30, 0.30), rng.uniform(0.35, 0.85)])
        jit = 0.25 if corrective else 0.10
        # Start DECOUPLED from the gate position (D34): spawning always
        # in front of the gate lets a policy pass training by flying
        # straight; independent spawns force genuine gate-seeking from
        # varied approach angles (the paper's OOD collapse on novel gate
        # positions suggests their data had this coupling too).
        start = np.array([rng.uniform(-0.95, 0.95),
                          rng.uniform(-1.9, -1.2),
                          rng.uniform(0.60, 1.00) + (0.15 if corrective else 0)])
        # Spawn facing the direction of flight (+y): the gate and the
        # hover target sit in the scene camera the whole run, as in the
        # paper. The PD yaw error is wrap-aware, so holding pi is safe.
        # slight initial-heading jitter (D34): the drone should not
        # always face exactly down the room
        yaw0 = np.pi + float(rng.uniform(-0.26, 0.26))
        self.reset_scene(bxy, start, gate_xy=gate_xy, yaw=yaw0, obj=obj)
        ex, frames = self.expert, []
        task = PROMPT_NAV.format(obj=obj)
        gate_z = 0.85
        ok = True
        crossed = {"v": False}
        d = self.data

        def track_cross():
            in_ap = (abs(d.qpos[0] - gate_xy[0]) < 0.45
                     and 0.34 + MAT_TOP < d.qpos[2] < 1.40 + MAT_TOP)
            if in_ap and d.qpos[1] > gate_xy[1]:
                crossed["v"] = True

        pre = np.array([gate_xy[0], gate_xy[1] - 0.55, gate_z])
        if corrective:
            # the paper's recovery flavour: pass near a random gate extremity
            side = rng.integers(4)
            off = [np.array([-0.30, 0, 0]), np.array([0.30, 0, 0]),
                   np.array([0, 0, 0.35]), np.array([0, 0, -0.35])][side]
            pre = pre + off
        ex.set_goal(xyz=pre, arm=ex.q_travel)
        ok &= self.run_until(frames, task,
                             lambda: np.linalg.norm(d.qpos[0:3] - ex.goal) < 0.05,
                             timeout_s=10.0)
        post = np.array([gate_xy[0], gate_xy[1] + 0.55, gate_z])
        ex.set_goal(xyz=post)
        for _ in range(int(6.0 * FPS)):
            self.step(frames, task)
            track_cross()
            if np.linalg.norm(d.qpos[0:3] - post) < 0.06:
                break
        # hover floor raised for the stand era (2026-08-26): the object
        # now tops out at ~0.53 on its pedestal, and a 0.55 body hover
        # put the legs INSIDE it; 0.78 keeps the legs 8+ cm clear
        hover = np.array([bxy[0], bxy[1], rng.uniform(0.78, 0.95)])
        ex.set_goal(xyz=hover)
        ok &= self.settle_near(frames, task, tol=0.05, timeout_s=12.0)
        self.run_until(frames, task, lambda: False, timeout_s=3.0)  # hold
        hov_ok = np.linalg.norm(d.qpos[0:2] - bxy) < 0.25
        return frames, {"task": task,
                        "kind": "corrective" if corrective else "nav",
                        "gate": bool(crossed["v"]), "hover": bool(hov_ok),
                        "success": bool(crossed["v"] and hov_ok)}


# ---------------------------------------------------------------------------
# collection entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--out_repo_id", required=True)
    ap.add_argument("--seed", type=int, default=71000)
    ap.add_argument("--mix", default="8,5,2",
                    help="manip,nav,corrective counts (must sum to --episodes)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {f"observation.images.{k}": {
        "dtype": "video", "shape": (IMG, IMG, 3),
        "names": ["height", "width", "channels"]} for k in CAMS}
    features["observation.state"] = {"dtype": "float32",
                                     "shape": (len(STATE_NAMES),),
                                     "names": STATE_NAMES}
    features["scene_state"] = {"dtype": "float32",
                                           "shape": (len(SCENE_NAMES),),
                                           "names": SCENE_NAMES}
    features["action"] = {"dtype": "float32",
                          "shape": (len(ACTION_NAMES),),
                          "names": ACTION_NAMES}
    enc = C.pick_rgb_encoder()
    enc.crf = 20
    dataset = LeRobotDataset.create(repo_id=args.out_repo_id, fps=FPS,
                                    features=features, robot_type="skygrip",
                                    rgb_encoder=enc)

    r = Runner(args.seed)
    counts = [int(x) for x in args.mix.split(",")]
    plan = (["manip"] * counts[0] + ["nav"] * counts[1]
            + ["corrective"] * counts[2]
            + ["descent"] * (counts[3] if len(counts) > 3 else 0))
    saved = attempts = 0
    names = list(r.objs)
    while saved < len(plan) and attempts < len(plan) * 8:
        kind = plan[saved]
        obj = names[saved % len(names)]   # alternate task objects
        attempts += 1
        if kind == "manip":
            frames, info = r.manip_episode(obj=obj)
        elif kind == "descent":
            frames, info = r.manip_episode(obj=obj, offset_hover=True)
        elif kind == "corrective" and saved % 2 == 0:
            # corrective slots alternate: edge-spawn block recovery / nav
            frames, info = r.manip_episode(obj="plush penguin",
                                           fallen_start=True)
        else:
            frames, info = r.nav_episode(
                corrective=(kind == "corrective"), obj=obj)
        # bank only CLEAN episodes (review 2026-08-19, episode 8: a
        # 1825-frame triple-retry marathon with a gate-failed marginal
        # pinch banked because it eventually placed): the final grasp
        # must have passed the alignment gate, and marathon episodes
        # (>1100 frames ~ 110 s) self-discard.
        clean = (info["success"] and info.get("gate", True)
                 and len(frames) <= 1400)
        if not clean:
            print(f"  attempt {attempts} [{kind}] DISCARDED "
                  f"({len(frames)} fr): {info}", flush=True)
            continue
        for f in frames:
            dataset.add_frame(f)
        dataset.save_episode(parallel_encoding=False)
        saved += 1
        print(f"episode {saved}/{len(plan)} [{kind}] banked "
              f"({len(frames)} frames) {info}", flush=True)
    print(f"DONE: {saved}/{len(plan)} banked in {attempts} attempts")


if __name__ == "__main__":
    main()
