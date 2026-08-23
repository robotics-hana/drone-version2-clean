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
        step = SP_STEP_CARRY if self.goal_grip < 0.5 else SP_STEP
        if self.slow:
            step = SP_STEP_FINAL
        d = np.clip(self.goal - self.sp, -step, step)
        self.sp = self.sp + d
        dg = np.clip(self.goal_grip - self.grip, -GRIP_STEP, GRIP_STEP)
        self.grip = self.grip + dg
        dyaw = float(np.clip(self.goal_yaw - self.yaw, -YAW_STEP, YAW_STEP))
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
            "weight": dict(body=self.wbody, adr=self.wadr,
                           aim_z=WEIGHT_AIM_Z, close=0.45,
                           narrow=0.75, gate_x=0.0045, gate_y=0.0065,
                           ap_lo=5.0, ap_hi=12.0, stage=0.04,
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
                              park=(-0.55, 1.05)),
        }
        self.cur = self.objs["weight"]
        self.mustard_adr = self.badr   # bottle = scenery only now
        self.bin_id = m.body("bin").id
        self.bin2_id = m.body("bin2").id
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

    # -- scene randomisation -------------------------------------------------
    def reset_scene(self, task_xy, drone_xyz, gate_xy=None,
                    bottle_mass=BOTTLE_MASS, yaw=0.0, obj="weight"):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        k = bottle_mass / self.base_mass
        m.body_mass[self.bot] = self.base_mass * k
        m.body_inertia[self.bot] = self.base_inertia * k
        # the episode's task object goes to the task spot; every other
        # object parks in-scene as background clutter
        self.cur = self.objs[obj]
        for key, o in self.objs.items():
            xy = (task_xy if key == obj
                  else np.asarray(o["park"])
                  + self.rng.uniform(-0.25, 0.25, 2))
            d.qpos[o["adr"]:o["adr"] + 2] = xy
            d.qpos[o["adr"] + 2] = MAT_TOP + 0.005
            # random yaw per episode (review 2026-08-20: rotated penguin
            # demos) -- the penguin faces a random direction; the grasp
            # is orientation-agnostic (spherical head pinch)
            t = self.rng.uniform(0, 2 * np.pi)
            d.qpos[o["adr"] + 3:o["adr"] + 7] = [np.cos(t / 2), 0, 0,
                                                 np.sin(t / 2)]
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
    def manip_episode(self, obj="weight", fallen_start=False):
        rng = self.rng
        # object spawn varies widely too (review 2026-08-19)
        bxy = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.20, 1.00)])
        # WIDE start variation (review 2026-08-19: "the start position of
        # the drone varies ... for all tasks")
        start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.6),
                          rng.uniform(0.55, 0.95)])
        self.reset_scene(bxy, start, obj=obj)
        ex, frames = self.expert, []
        task = PROMPT_MANIP.format(obj=obj)
        if fallen_start:
            # Corrective flavour (penguin era, D30): the object spawns
            # at the WORKSPACE EDGE, far outside the nominal region --
            # recovery-to-coverage (the ballasted penguin cannot tip
            # or roll).
            adr = self.cur["adr"]
            edge_x = float(rng.choice([-1, 1])) * rng.uniform(0.75, 1.0)
            self.data.qpos[adr:adr + 2] = [edge_x, rng.uniform(0.1, 1.1)]
            mujoco.mj_forward(self.model, self.data)
        ok = True
        self.run_until(frames, task, lambda: False, timeout_s=1.0)  # hover in

        welded, gate_ok, grasp_try = False, False, 0
        pre_close = np.zeros(3)
        first_pass = True
        vadr = self.model.jnt_dofadr[
            self.model.body_jntadr[self.cur["body"]]]
        for grasp_try in range(3):
            if grasp_try > 0:
                # wait for the object to be AT REST before re-aiming: a
                # knocked bottle keeps rolling, and an aim captured
                # mid-roll misses by ~20 cm once it stops (measured)
                self.run_until(frames, task,
                               lambda: float(np.linalg.norm(
                                   self.data.qvel[vadr:vadr + 6])) < 0.03,
                               timeout_s=5.0)
            # aim (and, for the penguin, an alignment yaw) from the
            # LIVE pose.
            aim, yaw_g = self.live_target()
            if yaw_g is not None and not first_pass:
                ex.set_goal(yaw=yaw_g)
            if first_pass:
                # travel above the object, arm at the camera-down pose
                over = aim - ex.off_travel + np.array([0, 0, 0.30])
                if yaw_g is not None:
                    # Penguin approach (user 2026-08-23): fly AT the
                    # object nose-first so it is in the forward view,
                    # brake 0.55 m short, slide to directly overhead,
                    # and only THEN apply the beak-alignment yaw.
                    # (Distinct from the REVERTED 2026-08-21 variant:
                    # no yaw-while-translating spiral -- the facing
                    # turn completes at a dead hover before the leg.)
                    here = self.data.qpos[0:2]
                    to = aim[0:2] - here
                    dist = float(np.linalg.norm(to))
                    u = to / max(1e-6, dist)
                    face = float(np.arctan2(u[0], -u[1]))
                    ex.set_goal(yaw=face, arm=ex.q_travel)
                    self.run_until(frames, task,
                                   lambda: abs(ex.yaw - ex.goal_yaw) < 0.03,
                                   timeout_s=8.0)
                    if dist > 0.60:
                        stand = over.copy()
                        stand[0:2] = aim[0:2] - 0.55 * u
                        ex.set_goal(xyz=stand)
                        ok &= self.settle_near(frames, task, tol=0.03)
                    # swing to the grasp arm pose HERE at the standoff
                    # (review 2026-08-23: the travel->grasp swing used
                    # to happen ABOVE the penguin, arcing the gripper
                    # out to its side and back before descending; with
                    # the swing done short of the object, the last leg
                    # is one straight slide-over + vertical descent)
                    ex.set_goal(arm=ex.q_grasp)
                    self.run_until(frames, task, lambda: False,
                                   timeout_s=1.5)
                else:
                    ex.set_goal(xyz=over, arm=ex.q_travel)
                    ok &= self.settle_near(frames, task)
                first_pass = False
            # Swing to the grasp pose AT ALTITUDE, then descend VERTICALLY
            # (D17: swinging while descending swept the pads through cap
            # height). For a fallen target, finish the reorienting yaw
            # turn up here too.
            # slide target is YAW-AWARE (review 2026-08-23, "the gripper
            # moves forward next to the penguin then jerks left"): the
            # FK offsets are BODY-frame vectors -- subtracting them
            # unrotated is only correct at yaw 0 (the weight's case).
            # At the penguin's grasp yaw the forward-reaching arm hangs
            # rotated, so the old target parked the jaws BESIDE the head
            # and the correction passes dragged them over in visible
            # jerks. Rotating the offset by the commanded yaw puts the
            # jaws dead over the head on arrival; the beak-alignment
            # turn runs DURING the slide-over.
            yaw_cmd = yaw_g if yaw_g is not None else 0.0
            cyw, syw = np.cos(yaw_cmd), np.sin(yaw_cmd)
            offw = np.array([cyw * ex.off_grasp[0] - syw * ex.off_grasp[1],
                             syw * ex.off_grasp[0] + cyw * ex.off_grasp[1],
                             ex.off_grasp[2]])
            at = aim - offw
            if yaw_g is not None:
                ex.set_goal(yaw=yaw_g)
            ex.set_goal(xyz=np.array([at[0], at[1], at[2] + 0.30]),
                        arm=ex.q_grasp)
            ok &= self.settle_near(frames, task, tol=0.03)
            if yaw_g is not None:
                self.run_until(frames, task,
                               lambda: abs(ex.yaw - ex.goal_yaw) < 0.02,
                               timeout_s=8.0)
            ex.set_goal(xyz=at + np.array([0, 0, self.cur["stage"]]))
            ok &= self.settle_near(frames, task)
            # ALL-AXIS correction +4 cm above the object (D19): open pads
            # clear of everything; kills the loaded FK/droop z-bias that
            # otherwise levers the bottle over at the cap's bottom edge.
            # Tracks the LIVE target so a rolled bottle is followed.
            for _ in range(3):
                resid = self.jaws() - (aim + np.array([0, 0, self.cur["stage"]]))
                if (np.linalg.norm(resid[0:2]) < 0.0015
                        and abs(resid[2]) < 0.003):
                    break
                ex.set_goal(xyz=ex.goal - resid)
                ok &= self.settle_near(frames, task)
            corr = ex.goal - (at + np.array([0, 0, self.cur["stage"]]))
            # Descend at NORMAL speed (D19: creep descents accumulate
            # loaded y-drift, measured +11 mm bimodal landings).
            # HARD-GATE the descent on LIVE jaw alignment: the tight
            # settles can time out under the loaded trim (their False
            # return was accumulated into `ok` but never gated anything),
            # and descents then launched from poses tens of mm off,
            # sweeping the open jaws through cap height -- the actual
            # knock mechanism, reshuffled by timing noise across builds.
            aligned_xy = self.run_until(
                frames, task,
                lambda: float(np.linalg.norm(
                    (self.jaws() - aim)[0:2])) < 0.010,
                timeout_s=8.0)
            if not aligned_xy:
                continue          # never descend misaligned; retry pass
            ex.set_goal(xyz=at + corr)
            descend_bad = False
            for _ in range(int(6.0 * FPS)):
                self.step(frames, task)
                exy = float(np.linalg.norm((self.jaws() - aim)[0:2]))
                if exy > 0.025:
                    descend_bad = True     # drifting toward a sweep:
                    break                  # abort upward immediately
                if (abs(float((self.jaws() - aim)[2])) < 0.005
                        and exy < 0.010):
                    break
            if descend_bad:
                ex.set_goal(xyz=at + corr + np.array([0, 0, 0.20]))
                self.settle_near(frames, task, tol=0.06, timeout_s=6.0)
                continue
            # Close as soon as the object sits within the gate tolerance;
            # correct only if actually outside it (D19: no pre-grip
            # shuffle -- the dance was the knock mechanism).
            # gate axes live in the GRIPPER frame: after a reorienting
            # yaw (fallen-bottle grasp) the closing axis is no longer
            # world-x, and testing world axes let up to 6.5 mm of
            # closing-axis error through the 2 mm gate (measured: the
            # pre-narrow then clips the lying cap and rolls the bottle)
            def gframe(rvec):
                cy, sy = np.cos(ex.yaw), np.sin(ex.yaw)
                return np.array([cy * rvec[0] + sy * rvec[1],
                                 -sy * rvec[0] + cy * rvec[1], rvec[2]])
            for _ in range(3):
                resid = self.jaws() - aim
                g = gframe(resid)
                if (abs(g[0]) < self.cur["gate_x"]
                        and abs(g[1]) < self.cur["gate_y"]
                        and abs(g[2]) < 0.004):
                    break
                ex.set_goal(xyz=ex.goal - resid)
                self.settle_near(frames, task, tol=0.010)
            # pre-narrow -> gate -> weld+close BACK-TO-BACK (D19: the old
            # gate->narrow->pause order left ~4 s of hover drift between
            # the alignment check and the weld capture)
            narrow = self.cur["narrow"]
            ex.set_goal(grip=narrow)
            self.run_until(frames, task,
                           lambda: abs(ex.grip - narrow) < 0.01,
                           timeout_s=0.8, min_hold=1)
            gx, gy = self.cur["gate_x"], self.cur["gate_y"]
            aligned = lambda: (abs(gframe(self.jaws() - aim)[0]) < gx
                               and abs(gframe(self.jaws() - aim)[1]) < gy
                               and (self.jaws() - aim)[2] > -0.006)
            gate_ok = self.run_until(frames, task, aligned, timeout_s=2.0,
                                     min_hold=1)
            pre_close = self.jaws() - aim
            close = self.cur["close"]     # object-width close
            if False:  # bottle-era fallen close path retired (D30)
                # FALLEN grasp: APERTURE-CONFIRMED pickup (review
                # 2026-08-19: "when the MuJoCo clamp no longer shuts we
                # know we have picked up the dropped mustard"). Close on
                # any plausible pose; if the clamp physically STOPS at
                # cap width, the cap is between the pads -- weld at the
                # seated pose and lift. A full shut means a miss: reopen
                # and retry from the live pose.
                welded = False
                # ANISOTROPIC plausibility in the gripper frame (same
                # lesson as the alignment gate): the lying-cap landing
                # error lives mostly ALONG the cap axis, where a deeper
                # or shallower seat is harmless -- the ep15 air-welds
                # were CROSS-axis offsets (bottle beside the jaws)
                gpc = gframe(pre_close)
                if (abs(gpc[0]) < 0.006 and abs(gpc[1]) < 0.015
                        and abs(gpc[2]) < 0.010):
                    i_pre = self.ctrl.mppi._i_pos.copy()
                    ex.set_goal(grip=close)
                    self.run_until(frames, task,
                                   lambda: abs(ex.grip - close) < 0.01,
                                   timeout_s=2.0)
                    self.run_until(frames, task, lambda: False,
                                   timeout_s=0.5)
                    # the cap must PHYSICALLY be between the jaws before
                    # welding: the aperture alone is circular here (the
                    # close commands cap width, so an air-close also
                    # stops at 11.2 mm) -- an air-weld carried the bottle
                    # floating beside the jaws (review, ep15). Verify
                    # against the live cap site.
                    cap_now, _ = self.live_target()
                    # 13 mm: the close CENTRES the cap a few mm (the
                    # pinch working), so a 10 mm check refused honest
                    # grasps; ep15's air-welds sat at 15+ mm
                    seated = (float(np.linalg.norm(
                        self.jaws() - cap_now)) < 0.013
                        and float(self.data.qpos[self.gadr]) > 0.008)
                    if seated:
                        welded = True
                        self.weld_grasp(True)   # datum = seated pose
                        self.ctrl.mppi._i_pos[:] = i_pre
                        break
                    ex.set_goal(grip=1.0)       # not seated: reopen
                    self.run_until(frames, task,
                                   lambda: ex.grip > 0.99,
                                   timeout_s=1.5, min_hold=1)
            else:
                # UPRIGHT grasp: validated pre-close-gated weld protocol
                # (D12/D14) -- close only on a verified pose (a blind
                # close on a missed pose shoves the object).
                welded = bool(np.linalg.norm(pre_close) < 0.008
                              and self.object_settled())
                if welded:
                    self.weld_grasp(True)
                    i_pre = self.ctrl.mppi._i_pos.copy()
                    ex.set_goal(grip=close)
                    self.run_until(frames, task,
                                   lambda: abs(ex.grip - close) < 0.01,
                                   timeout_s=2.0)
                    # post-close aperture sanity: the clamp must have
                    # stopped at the object's width -- anything outside
                    # the seated range is a cheaty weld (air, mid-tip,
                    # neck), so release it and retry
                    ap_now = float(self.data.qpos[self.gadr]) * 1000
                    if self.cur["ap_lo"] < ap_now < self.cur["ap_hi"]:
                        self.weld_grasp(True)       # datum re-capture
                        self.ctrl.mppi._i_pos[:] = i_pre
                        break
                    self.weld_grasp(False)
                    welded = False
            # Missed or knocked: REORIENT drone and gripper to the
            # target's live pose and try once more (review 2026-08-19:
            # "when the mustard falls, reorientate the gripper to the new
            # position of the cap and pick it up") -- reopen, ascend
            # clear; the next pass re-aims from wherever the object is.
            ex.set_goal(grip=1.0)
            self.run_until(frames, task, lambda: ex.grip > 0.99,
                           timeout_s=1.5, min_hold=1)
            ex.set_goal(xyz=np.array([float(self.data.qpos[0]),
                                      float(self.data.qpos[1]),
                                      float(aim[2]) + 0.45]))
            self.settle_near(frames, task, tol=0.06, timeout_s=8.0)
        ap = float(self.data.qpos[self.gadr])
        self._diag = {"ap_close_mm": round(ap * 1000, 1),
                      "welded": welded, "retried": grasp_try > 0}
        if welded:
            # grasp-confirmation dwell (review 2026-08-19): hold ~0.3 s
            # after the close before lifting, as on the real platform
            # where the pinch must be confirmed before committing
            self.run_until(frames, task, lambda: False, timeout_s=0.3)
        # lift; on a failed grasp, RE-GRASP once (the validated 90->96.7%
        # mechanism from the pedestal pipelines): reopen, re-correct against
        # the bottle's LIVE position, close again.
        # Two-stage lift: the payload transfers at lift-off and the PD's
        # thrust feed-forward lags (D5: ~65 mm transient at 100 g) -- rise
        # 12 cm and DWELL so the sag is absorbed near the floor, invisible,
        # then climb. Kills the "dips then recovers" artifact from the
        # first review sample.
        # Lift-off, sag dwell, TURN, then fly. No separate climb stage
        # and no position-settles here: the loaded cruise droop sits above
        # any tight settle tolerance, so settle-based climb stages just
        # burn their full timeouts hovering (measured ~21 s of dead hover
        # -- the "stays there for a prolonged time" review item). Instead:
        # rise 12 cm until the weight is measurably airborne, dwell 2.5 s
        # while the D5 transfer sag is absorbed near the floor, yaw the
        # nose (scene camera) onto the bin, then climb en route.
        ex.set_goal(xyz=at + corr + np.array([0, 0, 0.12]))
        self.run_until(frames, task,
                       lambda: float(self.data.qpos[self.cur["adr"] + 2]) > 0.04,
                       timeout_s=4.0)
        # Climb to hover altitude and STABILISE before turning (review
        # 2026-08-19: no turning straight off the pickup). The at-hover
        # check is altitude-threshold + near-zero vertical speed, not a
        # position tolerance -- the loaded droop deadlocks tight settles.
        hover_z = float(at[2] + corr[2]) + 0.28
        ex.set_goal(xyz=at + corr + np.array([0, 0, 0.45]))
        self.run_until(frames, task,
                       lambda: (self.data.qpos[2] > hover_z
                                and abs(self.data.qvel[2]) < 0.05),
                       timeout_s=12.0)
        self.run_until(frames, task, lambda: False, timeout_s=1.5)  # hover
        # nose is the body -y axis: at yaw 0 it points along world -y, so
        # facing a world direction (tx, ty) needs yaw = atan2(tx, -ty)
        here = self.data.qpos[0:2]
        yaw_des = float(np.arctan2(self.bin_xy[0] - here[0],
                                   -(self.bin_xy[1] - here[1])))
        ex.set_goal(yaw=yaw_des)
        self.run_until(frames, task,
                       lambda: abs(ex.yaw - ex.goal_yaw) < 0.02,
                       timeout_s=8.0)

        held = float(self.data.qpos[self.cur["adr"] + 2]) > 0.15
        # Bin approach: cruise with the arm TUCKED, extend it EN ROUTE
        # (review 2026-08-19, "only reach out the arm when at the box"):
        # the tucked cruise removes most of the payload pitch-moment lead
        # that made weight drops take 18-35 s, and the ~10 s arm swing
        # overlaps the ~10 s approach flight, so it costs no time. The
        # first leg is body-aimed and stops 0.45 m short (0.2 m of coming
        # arm reach + 0.25 m of brake margin); at the brake -- with the
        # arm arrived -- the payload hang and the residual loaded lead are
        # BOTH measured fresh, and the creep leg aims the payload dead at
        # the bin centre.
        dirn = self.bin_xy - self.data.qpos[0:2]
        dirn = dirn / max(1e-6, np.linalg.norm(dirn))
        # Fly to the bin EDGE with the arm still tucked -- extending the
        # arm mid-flight shifts the CoM while translating and the
        # attitude loop visibly chases it (review 2026-08-19: "the flight
        # looks unstable ... go to the side or edge, stay at hover, then
        # extend arm to avoid jerkiness").
        # 0.75 m short, not 0.45: extending the arm shifts the CoM and
        # the body drifts forward ~0.3 m while it happens ("overshooting
        # by the box", review) -- from 0.75 m out the drift stays in open
        # air and the final leg is one monotonic forward creep.
        ex.set_goal(xyz=np.array([*(self.bin_xy - 0.75 * dirn), 0.75]))
        self.settle_near(frames, task, tol=0.20, timeout_s=25.0)
        self.run_until(frames, task,
                       lambda: float(np.linalg.norm(
                           self.data.qvel[0:2])) < 0.03,
                       timeout_s=8.0)
        # dead hover at the edge: NOW extend the arm, stationary, and let
        # the platform re-trim before measuring anything
        ex.set_goal(arm=ex.q_carry)
        self.run_until(frames, task,
                       lambda: float(np.linalg.norm(
                           self.data.qpos[7:9] - ex.q_carry)) < 0.06,
                       timeout_s=14.0)
        self.run_until(frames, task,
                       lambda: float(np.linalg.norm(
                           self.data.qvel[0:2])) < 0.03,
                       timeout_s=6.0)
        hang = (self.data.qpos[self.cur["adr"]:self.cur["adr"] + 2]
                - self.data.qpos[0:2])
        lead = self.data.qpos[0:2] - ex.sp[0:2]
        goal_xy = self.bin_xy - hang - lead
        ex.slow = True
        ex.set_goal(xyz=np.array([goal_xy[0], goal_xy[1], 0.75]))
        # Transport budget sized to the distance at carry speed (1.3 m at
        # 0.12 m/s ~ 11 s): a 10 s budget was firing the RELEASE mid-flight
        # and bombing the lab with the bottle (measured: every end position
        # scattered along the mat->bin path). Release only after arrival.
        # Arrive and release on the PAYLOAD's position, not the body's.
        # Under load the PD trims to a steady offset from its waypoint
        # (measured: 14 cm behind, 7 cm low -- the integrator is gated
        # off until near-target, so the offset never closes) and any
        # body-position tolerance either deadlocks or releases off-bin.
        # A pilot aims the hanging payload: coarse-arrive, then correct
        # the goal by the weight's measured residual over the bin centre
        # (the same settled-residual discipline as the jaw corrections).
        # Wait for SP-arrival + stillness, NOT body-vs-goal proximity:
        # with lead compensation the body parks `lead` away from the goal
        # BY DESIGN (~0.28 m at 100 g), so a body-position settle can
        # never pass and burned its full timeout on every weight episode
        # (measured 18-35 s drop latency; the payload-residual loop below
        # is the real arrival criterion).
        self.run_until(frames, task,
                       lambda: (float(np.linalg.norm(ex.sp - ex.goal))
                                < 0.01
                                and float(np.linalg.norm(
                                    self.data.qvel[0:2])) < 0.04),
                       timeout_s=16.0)
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
            # Release LOW: dropping from carry height reads as THROWING,
            # and the unload pop (the PD's payload trim releasing -- the
            # counterpart of the D5 pickup sag) amplifies it. Descend so
            # the payload hangs ~10 cm above the rim, then let go.
            hang_z = float(self.data.qpos[2]
                           - self.data.qpos[self.cur["adr"] + 2])
            # payload releases ~22 cm above the rim (review 2026-08-19:
            # was 10 cm, raised so the unload wobble can't clip the box
            # walls while the drone restabilises; still far below the
            # carry-height release that read as throwing)
            ex.set_goal(xyz=np.array([float(ex.goal[0]), float(ex.goal[1]),
                                      MAT_TOP + 0.16 + 0.22 + hang_z]))
            self.run_until(frames, task,
                           lambda: (float(np.linalg.norm(
                                        ex.sp - ex.goal)) < 0.01
                                    and abs(float(
                                        self.data.qvel[2])) < 0.05),
                           timeout_s=8.0)
            self.weld_grasp(False)
            ex.set_goal(grip=1.0)
            self.run_until(frames, task, lambda: ex.grip > 0.99,
                           timeout_s=2.0, min_hold=1)
            self.run_until(frames, task, lambda: False, timeout_s=0.6)  # fall
        ex.slow = False
        ex.set_goal(xyz=np.array([self.bin_xy[0], self.bin_xy[1] - 0.5, 0.9]),
                    arm=ex.q_travel)
        self.settle_near(frames, task, tol=0.08, timeout_s=6.0)
        self.run_until(frames, task, lambda: False, timeout_s=1.0)  # hover out

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
        hover = np.array([bxy[0], bxy[1], rng.uniform(0.55, 0.80)])
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
            + ["corrective"] * counts[2])
    saved = attempts = 0
    names = list(r.objs)
    while saved < len(plan) and attempts < len(plan) * 8:
        kind = plan[saved]
        obj = names[saved % len(names)]   # alternate task objects
        attempts += 1
        if kind == "manip":
            frames, info = r.manip_episode(obj=obj)
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
                 and len(frames) <= 1100)
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
