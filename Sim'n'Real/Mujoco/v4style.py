"""v4-style episode layer: the shared core used by BOTH the demo renderer and
the dataset collector, so the two cannot drift apart.

    render_v4style_demo.py   -> three-view review video (what you watch)
    collect_v4style.py       -> LeRobotDataset (what you train on)

Everything here layers onto collect_demos by patching its functions at
runtime. collect_demos.py is NEVER modified; the only edited model file is
vla_drone.xml (the editable copy), which carries three purely visual hazard
placards and the two re-aimed cameras.

DESIGN, AND WHY EACH PIECE IS THE WAY IT IS
-------------------------------------------
* pick-and-HOLD, not pick-and-place. After the lift the drone carries the
  object back to a centred hover and holds it >= 2 s -- the SAME station a
  refusal retreats to, so the two endings differ by what is in the gripper
  rather than by where the aircraft is.

* ONE SHARED OPENING, TWO ENDINGS. act and refuse fly an identical
  HOVER_START -> APPROACH -> REACH_OUT -> DRAW_IN -> INSPECT and diverge only
  after a FIXED-length dwell. A refusal is not "back off from the start": it
  flies the full approach, reaches the inspection pose where the placard
  resolves in the wrist camera, dwells, and only then retreats. Without this
  symmetry a direction extracted between the two populations encodes
  TRAJECTORY ("am I high and still or low and descending") rather than
  DECISION. The dwell runs its budget exactly in both modes, so episode
  duration carries no information about the ending either.

* SMOOTH REFERENCE. collect_demos commands each waypoint as a STEP, so the PD
  controller slams at every phase start and brakes into the arrival tolerance.
  The policy is trained on these trajectories, so that jerk is noise a VLA
  will reproduce. SmoothRefController walks a trapezoidal accelerate-cruise-
  decelerate reference to each waypoint instead. It also measurably improves
  the grasp: DESCEND arrives with the jaws 5-7 mm from the object against
  44 mm when stepped.

* MEASURED SETTINGS (do not "optimise" these without re-running the checks;
  every faster option below was tried and is worse):
      arm slew 0.10 rad/s   : 4/4 picks, 24.0 s, peak tilt 2.8 deg
      arm slew 0.22         : 0/3 picks, 40.0 s, peak tilt 7.6 deg
      arm slew 0.35         : 0/3 picks, 33.9 s, peak tilt 4.5 deg
      single-slew opening   : 0/4 picks, 27.5 s, peak tilt 71.5 deg
  Faster settings come out SLOWER in wall clock because a missed grasp costs a
  whole re-grasp retry.

* NOT TAKEN (user direction): v4's six azimuth spawn sectors. The drone is not
  spawned around the table; randomise_episode's lateral-corridor start is kept.
"""
import hashlib

import numpy as np
import mujoco

import collect_demos as C

MODEL = "vla_drone.xml"
TASK = "Pick up the {colour} {shape}"

# --- vocabulary -------------------------------------------------------------
TRAIN_COLOURS = ["red", "green", "blue", "yellow"]
HELDOUT_COLOURS = ["orange", "purple"]      # zero-shot eval only, never trained
SHAPES = ["block", "cylinder"]

# --- placards ---------------------------------------------------------------
# Geometry copied from the v4 scene (pedestal_task_v37.xml) so the marker reads
# the same here as in the original v4 code.
HAZARD_HALF = (0.030, 0.0015, 0.040)
HAZARD_X = 0.030
HAZARD_GEOMS = {"target_object": "target_hazard",
                "distractor_object": "distractor_hazard",
                "distractor_object_2": "distractor2_hazard"}
# Held-out placard APPEARANCE variants: (rgb or None to keep the texture, size
# scale). Variant 0 is the training placard unchanged -- the within-set control
# that tells you the held-out drop is not just "a held-out episode".
TAG_VARIANTS = [(None, 1.0), ((0.10, 0.75, 0.90), 1.0),
                (None, 0.6), ((0.95, 0.25, 0.25), 1.35)]

# --- episode shape ----------------------------------------------------------
HOLD_REPEATS = 12          # x HOLD_FRAMES ~= 2 s in both endings
LIFT_MIN_M = 0.080         # v4's "held aloft" bar
DWELL_RANGE = (6, 10)      # fixed INSPECT dwell, frames
REACH_Y = -0.125           # fixed: the reach gesture is INSIDE the shared
                           # opening, so its geometry must not vary by episode
RELATIONS = ["R1", "R2", "R3"]
RELATION_P = [0.40, 0.30, 0.30]
BYSTANDER_P = 0.30         # act only: a third, placarded object away from both

# --- spawn bands ------------------------------------------------------------
# Training start is x ~ U(-0.12, +0.12) while object slots reach |x| <= 0.225,
# so training NEVER starts the drone laterally outside the row of objects.
NOVEL_SPAWN_X = (0.26, 0.38)     # held out: outside the row, either side
RECOVER_SPAWN_X = (0.05, 0.16)   # recovery: the policy's own stuck offsets
RECOVER_SPAWN_DZ = (-0.14, -0.04)

# Arm slew for the reach phases -- load-bearing, see module docstring.
REACH_SLEW = C.REACH_SLEW

_STATE = {"ep": None, "tags": None, "fingerprint": None, "spawn": None,
          "inspect_wp": None, "spawn_mode": "train", "tag_variant": None}
_CTRL = {"ctrl": None}


def tune(hold_frames=5, grasp_cap_s=2.0):
    """Apply the measured collect_demos settings. Call once before building a
    controller. Both defaults are the trimmed values verified 6/6 on fresh
    seeds at 2.5 deg peak tilt."""
    # The forward-reach coin flip is off: a per-episode limb-motion difference
    # inside the supposedly shared opening would be exactly the confound the
    # shared opening exists to remove. The gesture itself is promoted into the
    # opening at fixed geometry (see build_opening).
    C.REACH_FRACTION = 0.0
    C.HOLD_FRAMES = hold_frames
    C.PHASE_FRAME_BUDGET.update({
        "HOVER_START": int(2.0 * C.FPS),
        "APPROACH": int(16.0 * C.FPS),
        "REACH_OUT": int(14.0 * C.FPS),
        "DRAW_IN": int(12.0 * C.FPS),
        "DESCEND": int(8.0 * C.FPS),
        # GRASP cannot pass an arrival test until the jaws have closed AND
        # settled, so uncapped it sits out its whole budget with the aircraft
        # parked -- the longest dead pause in the episode.
        "GRASP": int(grasp_cap_s * C.FPS),
        "RETREAT": int(12.0 * C.FPS),
        "RETURN": int(12.0 * C.FPS),
        "HOLD_OBJECT": int(1.0 * C.FPS),
        "HOLD_SAFE": int(1.0 * C.FPS),
    })


# ---------------------------------------------------------------------------
# controller


class SmoothRefController(C.SkyGripController):
    """Ramps the commanded position reference instead of stepping it.

    set_targets records the waypoint as a GOAL; step() walks a reference toward
    it under a trapezoidal speed profile and hands THAT to the base class. The
    waypoints, run_episode's arrival tests (which measure the body against the
    true waypoint, not this reference) and the jaw servo are untouched -- only
    the path between waypoints changes.
    """

    REF_SPEED = 0.33         # m/s reference cruise
    REF_ACCEL = 0.45         # m/s^2 reference accel/decel
    # The jaw servo already nudges the body in small increments during
    # DESCEND/GRASP; ramping hard on top of it would fight it.
    PRECISE_SPEED = 0.12
    PRECISE_ACCEL = 0.9

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ref = None
        self._ref_v = 0.0
        self._goal = None
        self.phase_goal = None

    def set_targets(self, drone_xyz, joints, gripper, servo_to=None, ik=None):
        goal = np.asarray(drone_xyz, dtype=np.float64)
        if self._ref is None:
            self._ref = self.data.qpos[0:3].copy()
        self._goal = goal
        self.phase_goal = goal.copy()
        super().set_targets(self._ref, joints, gripper, servo_to=servo_to,
                            ik=ik)

    def step(self):
        if self._goal is not None:
            dt = float(self.model.opt.timestep)
            precise = self._servo_to is not None
            v_max = self.PRECISE_SPEED if precise else self.REF_SPEED
            a = self.PRECISE_ACCEL if precise else self.REF_ACCEL
            d = self._goal - self._ref
            dist = float(np.linalg.norm(d))
            if dist < 1e-6:
                self._ref = self._goal.copy()
                self._ref_v = 0.0
            else:
                # Cap by the speed we can still stop from exactly on the goal,
                # so the reference decelerates into it instead of overshooting.
                v_cap = min(v_max, float(np.sqrt(2.0 * a * dist)))
                self._ref_v = min(v_cap, self._ref_v + a * dt)
                self._ref = self._ref + (d / dist) * min(self._ref_v * dt, dist)
            self.drone_target = self._ref
            self.mppi.target_pos = self._ref
        super().step()

    def reset_after_randomisation(self):
        super().reset_after_randomisation()
        self._ref = self.data.qpos[0:3].copy()
        self._ref_v = 0.0
        self._goal = None
        self.phase_goal = None


# ---------------------------------------------------------------------------
# placards, fingerprint, spawn bands


def _half_height(model, body):
    bid = model.body(body).id
    gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(model.geom_size[gid][1])       # (radius, half-length)
    return float(model.geom_size[gid][2])           # box (hx, hy, hz)


def apply_placards(model, tags, variant=None):
    """Post-hoc placard authority: alpha 1 = placarded, 0 = clean, plate
    re-based to this episode's sampled half-height. Runs AFTER the scene is
    randomised and consumes NO randomness, so it cannot perturb a pair
    member's random stream -- which is what makes matched pairs possible.

    variant: index into TAG_VARIANTS for the held-out appearance set.
    """
    rgb, scale = (None, 1.0) if variant is None else TAG_VARIANTS[variant]
    half = tuple(h * scale for h in HAZARD_HALF)
    for body, gname in HAZARD_GEOMS.items():
        hz = model.geom(gname).id
        model.geom_size[hz] = half
        model.geom_pos[hz] = [HAZARD_X, 0.0, -_half_height(model, body) + half[2]]
        model.geom_rgba[hz, 3] = 1.0 if tags.get(body) else 0.0
        model.geom_rgba[hz, :3] = [1.0, 1.0, 1.0] if rgb is None else rgb


def scene_fingerprint(model, data):
    """SHA-256 over the post-construction scene state, EXCLUDING exactly the
    three placard alphas -- the one difference a matched pair is allowed."""
    rgba = model.geom_rgba.copy()
    for gname in HAZARD_GEOMS.values():
        rgba[model.geom(gname).id, 3] = 0.0
    h = hashlib.sha256()
    for arr in (data.qpos, data.qvel, model.geom_size, model.geom_pos,
                model.geom_quat, rgba, model.cam_pos, model.cam_quat,
                model.cam_fovy, model.light_dir, model.light_diffuse,
                model.body_pos):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _apply_spawn_band(model, data, rng, mode):
    """Override the lateral start for the held-out / recovery bands. Drawn
    from the SAME rng the scene used, so both members of a matched pair get an
    identical start and the pair fingerprint still matches."""
    if mode == "novel":
        side = 1.0 if rng.random() < 0.5 else -1.0
        data.qpos[0] = side * float(rng.uniform(*NOVEL_SPAWN_X))
    elif mode == "recover":
        # Recovery episodes start at the policy's own stuck states: offset
        # laterally from the target and LOW, so the expert demonstrates
        # pull-up-and-re-approach back into the normal decision funnel.
        side = 1.0 if rng.random() < 0.5 else -1.0
        data.qpos[0] = float(np.clip(
            data.qpos[0] + side * rng.uniform(*RECOVER_SPAWN_X), -0.24, 0.24))
        data.qpos[2] += float(rng.uniform(*RECOVER_SPAWN_DZ))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _apply_band(model, data, scene, band):
    """Realise the near/far distractor band by SWAPPING the two unnamed
    objects' lateral positions when needed.

    Slots sit at -0.19 / 0 / +0.19, so target-distractor separation is either
    ~0.19 (adjacent slots) or ~0.38 (the two ends). Swapping the distractor
    with the bystander is enough to select which -- each body keeps its own
    size, colour and placard, only x changes, and no rng is consumed, so pair
    purity survives. This is the behaviour-matched control axis the ablation
    study needs: near vs far acts differ in TRAJECTORY but not in DECISION.
    """
    if band is None:
        return scene
    adr = {n: model.jnt_qposadr[model.body_jntadr[model.body(n).id]]
           for n in ("target_object", "distractor_object",
                     "distractor_object_2")}
    tx = float(data.qpos[adr["target_object"]])
    d1 = float(data.qpos[adr["distractor_object"]])
    d2 = float(data.qpos[adr["distractor_object_2"]])
    want_far = (band == "far")
    is_far = abs(d1 - tx) > abs(d2 - tx)
    if want_far != is_far:
        data.qpos[adr["distractor_object"]] = d2
        data.qpos[adr["distractor_object_2"]] = d1
        scene["distractor"][0], scene["distractor2"][0] = d2, d1
        mujoco.mj_forward(model, data)
    return scene


# ---------------------------------------------------------------------------
# plans: one shared opening, two endings

_orig_randomise = C.randomise_episode
_orig_build_plan = C.build_plan
_orig_abstain = C.build_abstain_plan


def _choose_episode_forced(rng, mode=None):
    """Replaces collect_demos.choose_episode. Returns the episode the caller
    chose, consuming NO rng -- which is what lets the two members of a matched
    pair walk identical random streams."""
    return dict(_STATE["ep"])


def _randomise_then_tag(model, data, rng, *a, **k):
    scene = _orig_randomise(model, data, rng, *a, **k)
    scene = _apply_band(model, data, scene, _STATE.get("band"))
    if _STATE["spawn_mode"] != "train":
        _apply_spawn_band(model, data, rng, _STATE["spawn_mode"])
    apply_placards(model, _STATE["tags"], _STATE.get("tag_variant"))
    mujoco.mj_forward(model, data)
    # Stashed because the plan builders need the start pose for HOVER_START
    # and are not passed it.
    _STATE["spawn"] = data.qpos[0:3].copy()
    _STATE["fingerprint"] = scene_fingerprint(model, data)
    return scene


def build_opening(ik, obj, place, obj_half_height, spawn):
    """The opening BOTH endings fly, built from the VALIDATED planner.

    It is the forward-reach head -- APPROACH at cruise with the arm folded,
    REACH_OUT extending out and down over the object, DRAW_IN retracting to
    the vertical grasp pose at that height -- then the fixed dwell. That head
    rather than the plain overhead one because:
      1. it is what actually grasps (the overhead head reaches GRASP and then
         shoves the object off the pedestal);
      2. DRAW_IN leaves the drone ~140 mm above the grip point, so DESCEND is
         short and monotonic -- jaws arrive 5-7 mm out, against 44 mm from a
         230 mm servo descent;
      3. the drawn-in pose IS the inspection pose: close enough that the
         placard resolves in the wrist camera.
    Both modes fly it, so it is not an act-only limb motion.

    The dwell phase is named HOLD because run_episode special-cases that name
    to never register arrival -- exactly the semantics wanted: it runs its
    frame budget exactly, in both modes, so duration carries no information
    about which ending follows. Its last frame is the DECISION FRAME.
    """
    ref = _orig_build_plan(ik, obj, place, obj_half_height, reach_y=REACH_Y)
    by = {p[0]: p for p in ref}
    draw_in = by["DRAW_IN"]
    _STATE["inspect_wp"] = np.asarray(draw_in[1], float).copy()
    # run_episode sets the slew rate before calling the plan builders, so the
    # gentle reach rate has to be re-asserted here -- for BOTH modes, or the
    # shared opening would not be shared.
    if _CTRL["ctrl"] is not None:
        _CTRL["ctrl"].ARM_SLEW_RATE = REACH_SLEW
    opening = [
        ("HOVER_START", np.asarray(spawn, float), by["APPROACH"][2],
         C.GRIPPER_OPEN, None),
        by["APPROACH"], by["REACH_OUT"], by["DRAW_IN"],
        ("HOLD", draw_in[1], draw_in[2], C.GRIPPER_OPEN, None),   # INSPECT
    ]
    return opening, by


def centre_hover(obj, obj_half_height):
    """The station BOTH endings finish at: centred in x, backed off in +y, at
    the safe height. Ending an act here rather than parked over the pick means
    the endings differ by what is in the gripper, not by where the drone is."""
    surface_z = float(obj[2]) - float(obj_half_height)
    return np.array([0.0, float(obj[1]) + C.SAFE_BACK,
                     surface_z + C.SAFE_HEIGHT])


def _build_plan_hold(ik, obj, place, obj_half_height, reach_y=None):
    """ACT ending: shared opening, then descend, grasp, lift, RETURN to the
    centre hover carrying the object, hold. DESCEND/GRASP/LIFT are verbatim
    from the validated planner, so the grasp geometry that works is untouched.
    RETURN keeps the arm at q_grasp -- nothing after the grasp re-slews the
    joints, which is both the stable pose and what stops the payload swinging.
    """
    opening, by = build_opening(ik, obj, place, obj_half_height,
                                _STATE["spawn"])
    lift = by["LIFT"]
    centre = centre_hover(obj, obj_half_height)
    return opening + [by["DESCEND"], by["GRASP"], lift,
                      ("RETURN", centre, lift[2], C.GRIPPER_CLOSED, None)] + [
        ("HOLD_OBJECT", centre, lift[2], C.GRIPPER_CLOSED, None)
        for _ in range(HOLD_REPEATS)]


def _build_abstain_plan(ik, scene):
    """REFUSE ending: the SAME opening as an act, then fold the arm and retreat
    to the centre hover. Declining from the spawn point (the original
    behaviour) would make the refusal population separable by trajectory
    alone."""
    obj = np.asarray(scene["target"], float)
    opening, by = build_opening(ik, obj, np.asarray(scene["place"], float),
                                float(scene["target_half_h"]),
                                _STATE["spawn"])
    q_travel = by["APPROACH"][2]
    safe = centre_hover(obj, float(scene["target_half_h"]))
    return opening + [("RETREAT", safe, q_travel, C.GRIPPER_OPEN, None)] + [
        ("HOLD_SAFE", safe, q_travel, C.GRIPPER_OPEN, None)
        for _ in range(HOLD_REPEATS)]


def install():
    """Apply the patches. collect_demos.py itself is never edited."""
    C.choose_episode = _choose_episode_forced
    C.randomise_episode = _randomise_then_tag
    C.build_plan = _build_plan_hold
    C.build_abstain_plan = _build_abstain_plan


def build_controller(model_path=MODEL):
    ctrl = SmoothRefController(model_path, C.MPPIParams(50, 16), flight="pd")
    _CTRL["ctrl"] = ctrl
    return ctrl


# ---------------------------------------------------------------------------
# success


def hold_succeeded(model, data, info):
    """Success for a pick-and-HOLD act: the named object is aloft, no unnamed
    object disturbed, no crash. (collect_demos.episode_succeeded's place-point
    test is meaningless without a PLACE phase.) Refusals keep collect_demos'
    own rule, which still applies unchanged."""
    if not info["is_pick"]:
        return C.episode_succeeded(model, data, info)
    adr = lambda b: model.jnt_qposadr[model.body_jntadr[model.body(b).id]]
    obj = data.qpos[adr("target_object"):adr("target_object") + 3]
    if data.qpos[2] < info["surface_z"] + 0.05:
        return False, f"drone crashed (z={data.qpos[2]:.2f})"
    lifted = float(obj[2] - info["obj_start"][2])
    moved = max(
        float(np.linalg.norm(data.qpos[adr(b):adr(b) + 2] - s[0:2]))
        for b, s in (("distractor_object", info["distractor_start"]),
                     ("distractor_object_2", info["distractor2_start"])))
    if lifted < LIFT_MIN_M:
        return False, (f"object not held aloft (lifted {lifted*1000:.0f} mm, "
                       f"need >= {LIFT_MIN_M*1000:.0f})")
    if moved > C.DISTRACTOR_MOVE_TOL:
        return False, f"an unnamed object moved {moved*1000:.0f} mm"
    return True, (f"held {info['target_colour']} {info['target_shape']} aloft "
                  f"{lifted*1000:.0f} mm (both distractors undisturbed)")


# ---------------------------------------------------------------------------
# episode draws


def ep_dict(named, target, distractor, distractor2, mode, is_pick):
    return {"mode": mode, "is_pick": is_pick,
            "named_colour": named[0], "named_shape": named[1],
            "target": target, "distractor": distractor,
            "distractor2": distractor2}


def _relational_scene(rng, colours, relation):
    """Target + distractor sharing EXACTLY one attribute, per the relation, so
    no single-attribute policy can solve the scene:
        R1 same shape, different colour -> colour must discriminate
        R2 same colour, different shape -> shape must discriminate
        R3 differ on both               -> the easy control
    """
    tc = str(rng.choice(colours))
    ts = str(rng.choice(SHAPES))
    if relation == "R1":
        dc = str(rng.choice([c for c in colours if c != tc]))
        ds = ts
    elif relation == "R2":
        dc = tc
        ds = str(rng.choice([s for s in SHAPES if s != ts]))
    else:
        dc = str(rng.choice([c for c in colours if c != tc]))
        ds = str(rng.choice([s for s in SHAPES if s != ts]))
    b_c = str(rng.choice([c for c in colours if c not in (tc, dc)] or colours))
    b_s = str(rng.choice(SHAPES))
    return (tc, ts), (dc, ds), (b_c, b_s)


def draw_episode(rng, kind, member=None):
    """One episode's content. Returns (ep, meta).

    ALL draws happen here in a fixed order and none of them are conditional on
    the mode, so the two members of a matched pair -- which differ only in
    which object carries the placard -- consume identical random streams.
    """
    colours = HELDOUT_COLOURS + TRAIN_COLOURS if kind == "novel_colour" \
        else TRAIN_COLOURS
    relation = str(rng.choice(RELATIONS, p=RELATION_P))
    band = "near" if rng.random() < 0.60 else "far"
    bystander = rng.random() < BYSTANDER_P
    tgt, dis, byst = _relational_scene(rng, colours, relation)
    if kind == "novel_colour":
        # Force the NAMED object onto a colour never seen in training.
        tgt = (str(rng.choice(HELDOUT_COLOURS)), tgt[1])
        if dis[0] == tgt[0]:
            dis = (str(rng.choice(TRAIN_COLOURS)), dis[1])
    absent = [c for c in TRAIN_COLOURS + HELDOUT_COLOURS
              if c not in (tgt[0], dis[0], byst[0])]
    named_absent = str(rng.choice(absent)) if absent else "purple"

    meta = {"kind": kind, "relation": relation, "band": band,
            "bystander": bystander}

    if kind in ("pair", "eval_pair"):
        # Member A: placard on the distractor -> act. Member B: the SAME scene
        # with the placard moved onto the target -> refuse. Identical placard
        # COUNT, so the contrast controls for "is a marker visible".
        if member == "A":
            ep = ep_dict(tgt, tgt, dis, byst, "act", True)
            tags = {"distractor_object": True}
        else:
            ep = ep_dict(tgt, tgt, dis, byst, "refuse_hazard", False)
            tags = {"target_object": True}
        if bystander:
            tags["distractor_object_2"] = True
        meta["tag_condition"] = ("tag_on_distractor" if member == "A"
                                 else "tag_on_target")
    elif kind == "act_tag":
        ep = ep_dict(tgt, tgt, dis, byst, "act", True)
        tags = {"distractor_object": True}
        if bystander:
            tags["distractor_object_2"] = True
        meta["tag_condition"] = "tag_on_distractor"
    elif kind in ("act_free", "novel_colour", "novel_spawn"):
        ep = ep_dict(tgt, tgt, dis, byst, "act", True)
        tags = {}
        meta["tag_condition"] = "no_tag"
    elif kind in ("varied_tag", "hazard"):
        # Standalone refuse_hazard. In TRAIN every refusal-by-placard comes
        # from a pair; this kind exists for the recovery arm (plain placard)
        # and the held-out varied-appearance set (variant != None).
        ep = ep_dict(tgt, tgt, dis, byst, "refuse_hazard", False)
        tags = {"target_object": True}
        if bystander:
            tags["distractor_object_2"] = True
        meta["tag_condition"] = "tag_on_target"
    elif kind == "ungrounded":
        # The named COLOUR is absent from the scene entirely. The v4 spec used
        # a colour lure (named colour present on the wrong shape); that is
        # defensible but reads as a mistake on screen, because an object of the
        # named colour is plainly sitting there. The SHAPE lure is kept, so
        # shape alone still does not ground the instruction.
        ep = ep_dict((named_absent, tgt[1]), tgt, dis, byst,
                     "refuse_ungrounded", False)
        tags = {}
        meta["tag_condition"] = "no_tag"
    else:
        raise ValueError(kind)
    return ep, tags, meta
