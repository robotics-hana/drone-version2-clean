"""v4-style 5-episode demo: pick-and-HOLD, hazard placards, forced attribute
binding, and a matched counterfactual pair -- rendered three-view.

WHY THIS EXISTS RATHER THAN collect_v4.py
-----------------------------------------
The cluster's v4 layer (collect_v4.py + its collect_demos lineage) no longer
produces working ACT episodes: its own stock pilot scores 0/3 acts, with the
gate-1 close index at 40-43 frames against the spec's locked <= 15 and the
jaws closing 563-668 mm from the target. Refusals still pass, which is why
the breakage is easy to miss. That is a pre-existing regression in that
lineage, reproduced here before any of this was written, and unrelated to
the camera work.

So the v4 DESIGN is reimplemented on top of the local pipeline, which does
grasp reliably. Everything below layers onto collect_demos through function
patches applied at runtime -- collect_demos.py itself is not modified, and
neither is any XML except vla_drone.xml (the editable copy), which gained
three purely visual hazard placards.

WHY THE ARM EXTENSION IS SLOW (it is the biggest single cost, ~12 s)
The arm reconfigures at 0.10 rad/s through a large joint travel, in two
stages (REACH_OUT then DRAW_IN). Both the rate and the two-stage path were
tested against faster/simpler alternatives and both are load-bearing:
    reach gesture, 0.10 rad/s : 4/4 picks, 24.0 s, peak tilt 2.8 deg
    reach gesture, 0.22 rad/s : 0/3 picks, 40.0 s, peak tilt 7.6 deg
    reach gesture, 0.35 rad/s : 0/3 picks, 33.9 s, peak tilt 4.5 deg
    single slew (EXTEND+SEEK) : 0/4 picks, 27.5 s, peak tilt 71.5 deg
The single-slew variant swings Joint_2 through ~0.74 rad in one motion while
hovering, and the reaction torque lands on the axis the flight controller is
weakest about. Every "faster" option ends up SLOWER in wall clock, because a
missed grasp costs a whole re-grasp retry. --slew and --opening exist only to
re-run those checks.

WHAT IS v4 HERE
  * ending is pick-and-HOLD: no TRANSPORT / PLACE / RELEASE. After the lift
    the drone carries the object back to a centred hover and holds it there
    for >= 2 s -- the same station a refusal retreats to, so the two endings
    differ by what is in the gripper rather than by where the aircraft is.
  * hazard placards: a black/amber checker plate beside an object. Refusal is
    a property of CONTEXT (which object wears the placard), not of a
    hard-coded forbidden colour as in the v3 rule this replaces.
  * forced attribute binding: the distractor shares exactly ONE attribute
    with the target, so no single-attribute policy can solve the scene.
    R1 = same shape, different colour (colour must discriminate);
    R2 = same colour, different shape (shape must discriminate);
    R3 = both differ (the easy control).
  * matched counterfactual pair: two scenes identical in every respect --
    sizes, positions, colours, lighting, camera jitter, spawn, and the NUMBER
    of placards visible -- differing only in WHICH object carries the
    placard, and therefore only in the correct behaviour. Proved, not
    asserted: a SHA-256 over the full post-construction scene state
    (excluding exactly the three placard alphas) must match across members.

  * ONE SHARED OPENING, TWO ENDINGS. act and refuse fly an identical
    HOVER_START -> APPROACH -> REACH_OUT -> DRAW_IN -> INSPECT and diverge
    only after the dwell. A refusal is NOT "back off from the start": it flies the
    full approach, descends to an inspection pose where the placard resolves
    in the wrist camera, dwells there, and only then retreats. Without that
    symmetry a direction extracted between the populations would encode
    TRAJECTORY ("am I high and still or low and descending") rather than
    DECISION -- the flaw in the earlier design where refusals climbed away.
    The dwell runs a FIXED frame budget in both modes and neither exits
    early, so episode duration cannot separate the classes either.

WHAT IS DELIBERATELY NOT TAKEN (user direction)
  v4's six azimuth spawn sectors and per-sector heading. The drone is NOT
  spawned around the table. randomise_episode's existing lateral-corridor
  start is kept -- hover over a slot chosen independently of the target
  (20/40/40), heading fixed at 0 -- so the demonstrated approach stays the
  lateral / up-down / forward one currently in use.

SMOOTHNESS. collect_demos commands each phase waypoint as a STEP: the
reference jumps the moment a phase starts, the PD controller sees a large
error and slams, then decelerates into the arrival tolerance and dwells --
which is what makes the flight read as stop-start and, in places, violent.
Since the policy is trained on these trajectories, that jerk is not cosmetic:
it is noise the VLA will reproduce and integrate into drift. SmoothRefController
below therefore drives a MOVING reference along a trapezoidal
accelerate-cruise-decelerate profile toward each waypoint, so the commanded
setpoint is continuous and bounded in speed. The waypoints, arrival tests and
success criteria are untouched -- only the path the reference takes between
them.

collect_demos' own random choice between an overhead and a forward-reach
approach is OFF here (REACH_FRACTION = 0), because a per-episode coin flip
would put a limb-motion difference inside the supposedly shared opening. The
reach gesture itself is not discarded -- it is promoted INTO the shared
opening at a fixed geometry, flown identically by both modes, because it is
the path that actually grasps (see _shared_opening).

Run:
    python render_v4style_demo.py
"""
import argparse
import contextlib
import hashlib
import io

import numpy as np
import mujoco
import imageio.v2 as imageio

import collect_demos as C

try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT = _FONT_S = None
    for _p in ("C:/Windows/Fonts/arialbd.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            _FONT, _FONT_S = ImageFont.truetype(_p, 19), ImageFont.truetype(_p, 15)
            break
        except OSError:
            continue
    if _FONT is None:
        _FONT = _FONT_S = ImageFont.load_default()
except ImportError:
    Image = None

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../../Reports/v4style_demo_5ep.mp4")
ap.add_argument("--seed", type=int, default=20)
ap.add_argument("--distance", type=float, default=1.35)
ap.add_argument("--diag", type=int, default=0,
                help="trace N act episodes phase-by-phase and exit")
ap.add_argument("--nosmooth", action="store_true",
                help="diagnostic: step setpoints instead of ramping them")
ap.add_argument("--oldplan", action="store_true",
                help="diagnostic: no shared opening, original approach head")
ap.add_argument("--holdframes", type=int, default=5)
ap.add_argument("--slew", type=float, default=-1.0,
                help="arm slew rate for the reach phases (rad/s)")
ap.add_argument("--graspcap", type=float, default=2.0,
                help="GRASP phase budget, seconds")
ap.add_argument("--opening", choices=("reach", "extend"), default="reach",
                help="shared-opening arm path: reach gesture, or one slew")
ap.add_argument("--spawn", choices=("train", "novel"), default="train",
                help="'novel' = the held-out spawn band (see NOVEL_SPAWN_X)")
args = ap.parse_args()

MODEL = "vla_drone.xml"
TASK = "Pick up the {colour} {shape}"

# Placard geometry, copied from the v4 scene so the marker reads identically.
HAZARD_HALF = (0.030, 0.0015, 0.040)
HAZARD_X = 0.030
HAZARD_GEOMS = {"target_object": "target_hazard",
                "distractor_object": "distractor_hazard",
                "distractor_object_2": "distractor2_hazard"}
# Repeats of the hold waypoint. A phase exits once it has arrived and held
# HOLD_FRAMES frames, and the drone is already at the hold pose, so each
# repeat costs ~HOLD_FRAMES frames: 12 gives >= 2 s in both endings.
HOLD_REPEATS = 12
LIFT_MIN_M = 0.080                      # v4's "held aloft" bar
DWELL_RANGE = (6, 10)                   # fixed INSPECT dwell, frames
# Fixed, not drawn: the reach gesture is part of the SHARED opening, so its
# geometry must be identical across modes and across pair members. Mid of
# collect_demos' REACH_Y_RANGE.
REACH_Y = -0.125
INSPECT_RISE = 0.13        # inspect pose above the grasp point (extend variant)
# Arm slew for the reach phases. REACH_OUT + DRAW_IN are ~48% of an episode
# and the body is nearly stationary through both (it reaches its waypoint in
# ~1.5 s, then waits ~6 s for the arm), so this looks like the obvious thing
# to speed up. It is not: raising it was MEASURED and it fails.
#     0.10 rad/s : 3/3 picks, 25.9 s, peak tilt 2.1 deg
#     0.22 rad/s : 0/3 picks, 40.0 s, peak tilt 7.6 deg
#     0.35 rad/s : 0/3 picks, 33.9 s, peak tilt 4.5 deg (one crash at 132 deg
#                  in a companion run)
# The faster swing throws the body off during REACH_OUT/DRAW_IN, so the drone
# reaches the inspection pose ~40 mm out instead of ~5 mm, the grasp misses,
# and each re-grasp retry adds a whole descend-grasp-lift cycle -- which is why
# the "faster" settings produce LONGER episodes. collect_demos' comment that
# this rate is load-bearing is correct, and the ramped reference does not buy
# any slack on it. Leave at 0.10; --slew exists only to re-run that check.
REACH_SLEW = C.REACH_SLEW if args.slew < 0 else args.slew

# The forward-reach gesture is an act-only limb motion; flying it would break
# the shared opening (see module docstring).
C.REACH_FRACTION = 0.0
# Fewer frames parked at each arrival: the ramped reference already arrives
# gently, so a long dwell only adds the stop-start the approach is meant to
# lose. GRASP/RELEASE are unaffected -- they additionally wait on
# gripper_settled(), so they cannot be cut short by this.
C.HOLD_FRAMES = args.holdframes
# Ramped travel takes longer in wall-clock frames than a step command, so the
# transit budgets have to be generous or a phase times out mid-flight.
C.PHASE_FRAME_BUDGET.update({
    "HOVER_START": int(2.0 * C.FPS),
    "APPROACH": int(16.0 * C.FPS),
    "EXTEND": int(14.0 * C.FPS),
    "SEEK": int(10.0 * C.FPS),
    "RETURN": int(12.0 * C.FPS),
    "REACH_OUT": int(14.0 * C.FPS),
    "DRAW_IN": int(12.0 * C.FPS),
    "DESCEND": int(8.0 * C.FPS),
    # GRASP has no arrival test it can pass until the jaws have both closed and
    # settled, so it otherwise sits out its whole budget with the aircraft
    # parked -- the longest dead pause in the episode. Capped just above the
    # measured close time instead.
    "GRASP": int(args.graspcap * C.FPS),
    "RETREAT": int(12.0 * C.FPS),
    "HOLD_OBJECT": int(1.0 * C.FPS),
    "HOLD_SAFE": int(1.0 * C.FPS),
})

# ---------------------------------------------------------------------------
# v4 layer, applied to collect_demos by patching -- its file is not touched.

_STATE = {"ep": None, "tags": None, "fingerprint": None}


def _choose_episode_forced(rng, mode=None):
    """Replaces collect_demos.choose_episode. Returns the episode this demo
    chose, consuming NO rng -- which is what lets the two members of a
    counterfactual pair walk identical random streams."""
    return dict(_STATE["ep"])


def _half_height(model, body):
    bid = model.body(body).id
    gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(model.geom_size[gid][1])        # (radius, half-length)
    return float(model.geom_size[gid][2])            # box (hx, hy, hz)


def apply_placards(model, tags):
    """Post-hoc placard authority: alpha 1 = placarded, 0 = clean, and the
    plate re-based to whatever half-height this episode sampled. Runs AFTER
    the scene is randomised and consumes no randomness, so it cannot perturb
    a pair member's stream."""
    for body, gname in HAZARD_GEOMS.items():
        hz = model.geom(gname).id
        model.geom_size[hz] = HAZARD_HALF
        model.geom_pos[hz] = [HAZARD_X, 0.0,
                              -_half_height(model, body) + HAZARD_HALF[2]]
        model.geom_rgba[hz, 3] = 1.0 if tags.get(body) else 0.0


def scene_fingerprint(model, data):
    """SHA-256 over the post-construction scene state, excluding exactly the
    three placard alphas -- the one difference a pair is allowed to have."""
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


_orig_randomise = C.randomise_episode


# --- held-out spawn axis -----------------------------------------------------
# The v4 spec's novel-spawn axis was a radial band beyond the training radius,
# which only means anything with the six azimuth sectors we do not use. The
# equivalent for the lateral-corridor spawn is a LATERAL start outside the
# training band, and the numbers make the choice for us:
#
#   training start x : U(-0.12, +0.12)          (randomise_episode)
#   object slots x   : -0.19 / 0 / +0.19, each jittered +-0.035 -> |x| <= 0.225
#
# So in training the drone ALWAYS starts inside the middle 24 cm -- never
# laterally outside the row of objects, and never further from a target than
# ~0.35 m. The held-out band starts it OUTSIDE the row, on either side:
NOVEL_SPAWN_X = (0.26, 0.38)
# 0.26 clears the outermost object a slot can reach (0.225) by 35 mm, so a
# novel start is unambiguously beyond the row rather than merely near its edge;
# 0.38 keeps the traverse inside the flight budget (worst case is a start at
# +0.38 with the target at -0.225, a 0.60 m lateral run).
#
# ONE factor changes: depth (y) and altitude (z) keep their training
# distributions untouched, so any generalisation gap is attributable to the
# lateral start alone rather than to a compound shift. This is the axis worth
# testing because the lateral corridor is the entire point of the spawn design
# -- the policy has to translate sideways to the NAMED object rather than grab
# whatever is beneath it -- and this band approaches the row from OUTSIDE it,
# which training never does.
#
# It shifts the traverse distribution up rather than guaranteeing a longer one
# (measured over 8 episodes: 5-53 cm here against 1-27 cm for training) --
# a novel start on the same side as an outer target can still be a short run.
# That is the honest description; do not claim every held-out episode is a
# longer traverse.
#
# MEASURED USABLE: 8/8 expert picks from this band, peak tilt 2.6 deg against
# 2.7 deg for training starts. So a held-out failure is the policy failing to
# generalise, not the expert being unable to fly the scene.


def _apply_novel_spawn(model, data, rng):
    """Held-out start: outside the object row, on a random side. Drawn from
    the SAME rng the scene used, so both members of a matched pair get an
    identical start and the pair fingerprint still matches."""
    side = 1.0 if rng.random() < 0.5 else -1.0
    data.qpos[0] = side * float(rng.uniform(*NOVEL_SPAWN_X))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _randomise_then_tag(model, data, rng, *a, **k):
    scene = _orig_randomise(model, data, rng, *a, **k)
    if args.spawn == "novel":
        _apply_novel_spawn(model, data, rng)
    apply_placards(model, _STATE["tags"])
    mujoco.mj_forward(model, data)
    # The lateral-corridor spawn randomise_episode just set. Stashed because
    # the plan builders need it for HOVER_START and are not passed it.
    _STATE["spawn"] = data.qpos[0:3].copy()
    _STATE["fingerprint"] = scene_fingerprint(model, data)
    return scene


class SmoothRefController(C.SkyGripController):
    """SkyGripController that ramps the commanded position reference instead
    of stepping it.

    collect_demos issues each waypoint as an instantaneous jump, so the PD
    controller starts every phase against a large error and ends it braking
    into the arrival tolerance. Here set_targets records the waypoint as a
    GOAL and step() walks a reference point toward it under a trapezoidal
    speed profile -- accelerate at REF_ACCEL, cruise no faster than
    REF_SPEED, and decelerate so the reference arrives with zero velocity.
    The controller therefore always tracks a nearby, continuously moving
    setpoint, which is what makes the flight smooth.

    Only the path BETWEEN waypoints changes. The waypoints themselves, the
    arrival tests in run_episode (which measure the body against the true
    waypoint, not this reference) and the jaw servo are untouched.
    """

    REF_SPEED = 0.33         # m/s, reference cruise speed
    REF_ACCEL = 0.45         # m/s^2, reference accel/decel
    # The jaw servo already moves the body in small increments during
    # DESCEND/GRASP; ramping on top of it would fight it, so precise phases
    # get a faster profile that effectively passes the command through.
    PRECISE_SPEED = 0.12
    PRECISE_ACCEL = 0.9

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ref = None
        self._ref_v = 0.0
        self._goal = None
        self.phase_goal = None       # waypoint of the phase now running

    def set_targets(self, drone_xyz, joints, gripper, servo_to=None, ik=None):
        goal = np.asarray(drone_xyz, dtype=np.float64)
        if self._ref is None:
            self._ref = self.data.qpos[0:3].copy()
        self._goal = goal
        self.phase_goal = goal.copy()
        # Hand the CURRENT reference to the base class, not the goal: the jump
        # is what we are removing. step() advances it from here.
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
                # Cap by the speed from which we can still stop exactly on the
                # goal (v = sqrt(2*a*s)), so the reference decelerates into it
                # instead of overshooting and being dragged back.
                v_cap = min(v_max, float(np.sqrt(2.0 * a * dist)))
                self._ref_v = min(v_cap, self._ref_v + a * dt)
                step_len = min(self._ref_v * dt, dist)
                self._ref = self._ref + (d / dist) * step_len
            self.drone_target = self._ref
            self.mppi.target_pos = self._ref
        super().step()

    def reset_after_randomisation(self):
        super().reset_after_randomisation()
        self._ref = self.data.qpos[0:3].copy()
        self._ref_v = 0.0
        self._goal = None
        self.phase_goal = None


def _shared_opening(ik, obj, place, obj_half_height, spawn):
    """The opening BOTH endings fly, built once from the VALIDATED planner so
    the two populations cannot drift apart.

    It is the forward-reach head -- APPROACH at cruise with the arm folded,
    REACH_OUT extending the arm out and down over the object, DRAW_IN
    retracting to the vertical grasp pose at that same height -- followed by
    the fixed dwell. Three reasons that head rather than the plain overhead
    one:

      1. It is what actually grasps. Measured over a 2x2 of plan x setpoint
         handling, the overhead head reaches GRASP with the object still in
         place and then shoves it off the pedestal (object ends on the floor,
         "lifted -358 mm"); the reach head is the path every successful pick
         in this project has flown.
      2. DRAW_IN leaves the drone ~140 mm above the grip point, so DESCEND is
         a short, monotonic move rather than a 230 mm servo descent from
         cruise -- DESCEND arrives with the jaws 7 mm from the object here
         against 44 mm for the long version.
      3. It gives the inspection pose for free: the drawn-in waypoint is close
         enough that the placard resolves in the wrist camera, which is the
         whole point of INSPECT.

    The gesture is flown by BOTH modes, so it is not an act-only limb motion
    -- which is exactly what the shared-opening requirement forbids. reach_y
    is a constant, so the two members of a pair fly identical geometry.

    The dwell phase is named HOLD because run_episode special-cases that name
    to never register arrival -- exactly the semantics wanted: it runs its
    frame budget exactly, in both modes, so duration carries no information
    about which ending follows. The last frame of it is the decision frame.
    """
    if args.opening == "extend":
        # Single-slew variant: the arm goes straight from the transit pose to
        # the grasp pose ONCE (EXTEND), at cruise, and the body then descends
        # to the inspection pose with the arm static (SEEK). Half the arm
        # travel of the reach gesture, hence roughly half the slew time.
        ref = _orig_build_plan(ik, obj, place, obj_half_height, reach_y=None)
        by = {p[0]: p for p in ref}
        at_inspect = (np.asarray(by["DESCEND"][1], float)
                      + np.array([0.0, 0.0, INSPECT_RISE]))
        _STATE["inspect_wp"] = at_inspect.copy()
        ctrl.ARM_SLEW_RATE = REACH_SLEW
        q_grasp = by["EXTEND"][2]
        opening = [
            ("HOVER_START", np.asarray(_STATE["spawn"], float),
             by["APPROACH"][2], C.GRIPPER_OPEN, None),
            by["APPROACH"], by["EXTEND"],
            ("SEEK", at_inspect, q_grasp, C.GRIPPER_OPEN, None),
            ("HOLD", at_inspect, q_grasp, C.GRIPPER_OPEN, None),   # INSPECT
        ]
        return opening, by

    ref = _orig_build_plan(ik, obj, place, obj_half_height, reach_y=REACH_Y)
    by = {p[0]: p for p in ref}
    draw_in = by["DRAW_IN"]
    _STATE["inspect_wp"] = np.asarray(draw_in[1], float).copy()
    # The reach gesture swings the arm through a large angle; the validated
    # path slews it gently (REACH_SLEW) so the reaction torque does not shove
    # the body. run_episode sets the rate before calling the plan builders, so
    # this has to be re-asserted here -- and for BOTH modes, or the shared
    # opening would not be shared.
    ctrl.ARM_SLEW_RATE = REACH_SLEW
    opening = [
        ("HOVER_START", np.asarray(spawn, float), by["APPROACH"][2],
         C.GRIPPER_OPEN, None),
        by["APPROACH"], by["REACH_OUT"], by["DRAW_IN"],
        ("HOLD", draw_in[1], draw_in[2], C.GRIPPER_OPEN, None),   # INSPECT
    ]
    return opening, by


_orig_build_plan = C.build_plan


def _centre_hover(obj, obj_half_height):
    """The station both endings finish at: centred in x over the work area,
    backed off in +y, at the safe height. Ending an act here rather than
    parked directly over the pick means the two endings differ by what is IN
    THE GRIPPER and not by where the aircraft is."""
    surface_z = float(obj[2]) - float(obj_half_height)
    return np.array([0.0, float(obj[1]) + C.SAFE_BACK,
                     surface_z + C.SAFE_HEIGHT])


def _build_plan_hold(ik, obj, place, obj_half_height, reach_y=None):
    """ACT ending: shared opening, then descend, grasp, lift, RETURN to the
    centre hover carrying the object, and hold it there.

    DESCEND/GRASP/LIFT are taken verbatim from the validated planner, so the
    grasp geometry that actually works is preserved untouched. RETURN keeps
    the arm at q_grasp -- nothing after the grasp re-slews the joints, which
    is both the stable pose (mass hanging below the thrust point) and what
    stops the payload swinging."""
    opening, by = _shared_opening(ik, obj, place, obj_half_height,
                                  _STATE["spawn"])
    lift = by["LIFT"]
    centre = _centre_hover(obj, obj_half_height)
    return opening + [by["DESCEND"], by["GRASP"], lift,
                      ("RETURN", centre, lift[2], C.GRIPPER_CLOSED, None)] + [
        ("HOLD_OBJECT", centre, lift[2], C.GRIPPER_CLOSED, None)
        for _ in range(HOLD_REPEATS)]


_orig_abstain = C.build_abstain_plan


def _build_abstain_plan_v4(ik, scene):
    """REFUSE ending: the SAME opening as an act -- fly across, reach out over
    the object, draw in to the grasp pose, dwell at the inspection pose -- and
    only then fold the arm and retreat to the safe hover. Declining from the
    spawn point (the original behaviour) would make the refusal population
    separable by trajectory alone, so the extracted direction would encode
    "am I high and still or low and descending" rather than the decision."""
    obj = np.asarray(scene["target"], float)
    opening, by = _shared_opening(ik, obj, np.asarray(scene["place"], float),
                                  float(scene["target_half_h"]),
                                  _STATE["spawn"])
    q_travel = by["APPROACH"][2]
    safe = _centre_hover(obj, float(scene["target_half_h"]))
    return opening + [("RETREAT", safe, q_travel, C.GRIPPER_OPEN, None)] + [
        ("HOLD_SAFE", safe, q_travel, C.GRIPPER_OPEN, None)
        for _ in range(HOLD_REPEATS)]


def _build_plan_no_opening(ik, obj, place, obj_half_height, reach_y=None):
    """Diagnostic only: the pre-shared-opening plan (original approach head,
    place tail dropped, hold appended)."""
    plan = _orig_build_plan(ik, obj, place, obj_half_height, reach_y=reach_y)
    keep = [p for p in plan if p[0] not in ("TRANSPORT", "PLACE", "RELEASE")]
    lift = next(p for p in keep if p[0] == "LIFT")
    return keep + [("HOLD_OBJECT", lift[1], lift[2], C.GRIPPER_CLOSED, None)
                   for _ in range(HOLD_REPEATS)]


C.choose_episode = _choose_episode_forced
C.randomise_episode = _randomise_then_tag
C.build_plan = _build_plan_no_opening if args.oldplan else _build_plan_hold
if not args.oldplan:
    C.build_abstain_plan = _build_abstain_plan_v4


def hold_succeeded(model, data, info):
    """Success for a pick-and-hold act episode: the named object is aloft, no
    unnamed object was disturbed, no crash. (The place-point test in
    collect_demos.episode_succeeded is meaningless without a PLACE phase.)
    Refusals are judged by collect_demos' own rule, which still applies."""
    if not info["is_pick"]:
        return C.episode_succeeded(model, data, info)
    adr = lambda b: model.jnt_qposadr[model.body_jntadr[model.body(b).id]]
    obj = data.qpos[adr("target_object"):adr("target_object") + 3]
    surface = info["surface_z"]
    if data.qpos[2] < surface + 0.05:
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
# scene setup

_CTRL_CLS = C.SkyGripController if args.nosmooth else SmoothRefController
with contextlib.redirect_stdout(io.StringIO()):
    ctrl = _CTRL_CLS(MODEL, C.MPPIParams(50, 16), flight="pd")
model, data = ctrl.model, ctrl.data
ik = C.ArmIK(model)
rend_front = mujoco.Renderer(model, height=480, width=640)
rend_small = mujoco.Renderer(model, height=240, width=320)
# Segmentation renderer, used only to measure how many pixels of the hazard
# placard the wrist camera actually resolves at the decision frame.
seg = mujoco.Renderer(model, height=240, width=320)
seg.enable_segmentation_rendering()

ped = model.body("pedestal").id
ped_gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ped][0]

_PINNED = {}
for _n in ("wrist_cam", "scene_cam"):
    _c = model.camera(_n).id
    _PINNED[_c] = (model.cam_pos[_c].copy(), model.cam_quat[_c].copy())


def pin_cams():
    for _c, (p, q) in _PINNED.items():
        model.cam_pos[_c] = p
        model.cam_quat[_c] = q


front = mujoco.MjvCamera()
front.azimuth, front.elevation, front.distance = 90, -14, args.distance


def ep_dict(named, target, distractor, distractor2, mode, is_pick):
    return {"mode": mode, "is_pick": is_pick,
            "named_colour": named[0], "named_shape": named[1],
            "target": target, "distractor": distractor,
            "distractor2": distractor2}


def placard_text(tags):
    on = [w for b, w in (("target_object", "NAMED OBJECT"),
                         ("distractor_object", "distractor"),
                         ("distractor_object_2", "bystander")) if tags.get(b)]
    return "placard on " + " + ".join(on) if on else "no placard in scene"


def placard_pixels():
    """Placard area, in wrist-camera pixels, right now."""
    seg.update_scene(data, camera="wrist_cam")
    s = seg.render()
    G = int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = [model.geom(g).id for g in HAZARD_GEOMS.values()
           if model.geom_rgba[model.geom(g).id, 3] > 0.5]
    return int(sum(((s[:, :, 1] == G) & (s[:, :, 0] == g)).sum() for g in ids))


def run_episode(ep, tags, seed):
    """One v4-style episode. Returns (frames, ok, why, fingerprint, px)."""
    _STATE["ep"], _STATE["tags"] = ep, tags
    rng = np.random.default_rng(seed)
    # Fixed-length INSPECT dwell, drawn per episode and applied to the phase
    # run_episode treats as never-arriving, so it burns exactly this budget.
    C.PHASE_FRAME_BUDGET["HOLD"] = int(rng.integers(DWELL_RANGE[0],
                                                    DWELL_RANGE[1] + 1))
    frames, dwell_px = [], []

    def grab():
        pin_cams()
        surface_z = model.body_pos[ped][2] + model.geom_size[ped_gid, 2]
        front.lookat[:] = (model.body_pos[ped][0], model.body_pos[ped][1],
                           surface_z + 0.15)
        rend_front.update_scene(data, camera=front)
        tf = rend_front.render().copy()
        rend_small.update_scene(data, camera="wrist_cam")
        tw = rend_small.render().copy()
        rend_small.update_scene(data, camera="scene_cam")
        ts = rend_small.render().copy()
        frames.append(np.hstack([tf, np.vstack([tw, ts])]))
        # While the controller is working the inspection waypoint, log what the
        # wrist camera resolves of the placard; the LAST such frame is the
        # decision frame, so the last value is the decision-frame figure.
        wp = _STATE.get("inspect_wp")
        if wp is not None and ctrl.phase_goal is not None and \
                np.allclose(ctrl.phase_goal, wp):
            dwell_px.append(placard_pixels())

    with contextlib.redirect_stdout(io.StringIO()):
        _, info = C.run_episode(model, data, None, ctrl, ik, rng, TASK,
                                mode=ep["mode"], on_frame=grab)
        ok, why = hold_succeeded(model, data, info)
    px = dwell_px[-1] if dwell_px else 0
    return frames, ok, why, _STATE["fingerprint"], px


ACT_C, REF_C = (60, 190, 110, 255), (215, 70, 70, 255)


def annotate(img, lines, accent):
    if Image is None:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, 640, 6], fill=accent)
    y = 12
    for text, font in lines:
        # Keep the caption inside the front-view panel: a line long enough to
        # run past 640 px would otherwise print over the camera tiles.
        while text and dr.textlength(text, font=font) > 612:
            text = text[:-2]
        w = dr.textlength(text, font=font)
        dr.rectangle([8, y, 8 + w + 12, y + font.size + 9], fill=(0, 0, 0, 170))
        dr.text((14, y + 4), text, font=font, fill=(255, 255, 255))
        y += font.size + 13
    for lab, (x, yy) in (("wrist_cam", (648, 8)), ("scene_cam", (648, 248))):
        w = dr.textlength(lab, font=_FONT_S)
        dr.rectangle([x, yy, x + w + 12, yy + 24], fill=(0, 0, 0, 150))
        dr.text((x + 6, yy + 4), lab, font=_FONT_S, fill=(255, 255, 255))
    return np.asarray(im)


# --- the five episodes -----------------------------------------------------
# Pair members 1 and 2 are identical in every respect except which object
# wears the placard: same colours, shapes, instruction, and placard COUNT.
PAIR_TARGET = ("green", "block")
PAIR_DISTRACTOR = ("blue", "block")          # R1: same shape, colour decides
PAIR_BYSTANDER = ("yellow", "cylinder")

EPISODES = [
    dict(idx=1, label="pair A", relation="R1",
         ep=ep_dict(PAIR_TARGET, PAIR_TARGET, PAIR_DISTRACTOR, PAIR_BYSTANDER,
                    "act", True),
         tags={"distractor_object": True}),
    dict(idx=2, label="pair B", relation="R1",
         ep=ep_dict(PAIR_TARGET, PAIR_TARGET, PAIR_DISTRACTOR, PAIR_BYSTANDER,
                    "refuse_hazard", False),
         tags={"target_object": True}),
    dict(idx=3, label="", relation="R1",
         ep=ep_dict(("blue", "cylinder"), ("blue", "cylinder"),
                    ("red", "cylinder"), ("yellow", "block"), "act", True),
         tags={}),
    dict(idx=4, label="tagged bystander", relation="R2",
         ep=ep_dict(("yellow", "block"), ("yellow", "block"),
                    ("yellow", "cylinder"), ("blue", "block"), "act", True),
         tags={"distractor_object_2": True}),
    # The named COLOUR appears nowhere in the scene. An earlier version put a
    # red block in a scene whose instruction named a red cylinder -- correct
    # under the v4 spec's lure design (the named colour+shape PAIR is absent,
    # each attribute present alone, so refusing requires joint binding) but it
    # reads as a mistake on screen, because a red object is plainly sitting
    # there. The shape lure is kept (there IS a cylinder), so shape alone still
    # does not ground the instruction.
    dict(idx=5, label="colour absent from scene", relation="shape lure",
         ep=ep_dict(("purple", "cylinder"), ("green", "block"),
                    ("blue", "cylinder"), ("yellow", "block"),
                    "refuse_ungrounded", False),
         tags={}),
]

if args.diag:
    # Phase-by-phase trace of act episodes -- no video, no retries. Reports
    # total frames and PEAK BODY TILT, so a trim can be checked against the
    # attitude it costs rather than eyeballed.
    print(f"[cfg] slew={REACH_SLEW} grasp_cap={args.graspcap}s "
          f"hold_frames={args.holdframes}", flush=True)
    _tot, _tilt = [], []
    for _k in range(args.diag):
        _STATE["ep"] = EPISODES[0]["ep"]
        _STATE["tags"] = EPISODES[0]["tags"]
        _rng = np.random.default_rng(args.seed + _k)
        C.PHASE_FRAME_BUDGET["HOLD"] = 8
        _peak = [0.0]

        def _watch():
            _peak[0] = max(_peak[0],
                           abs(np.degrees(C._rp(data, 0))),
                           abs(np.degrees(C._rp(data, 1))))

        print(f"--- act episode, seed {args.seed + _k} "
              f"(spawn={args.spawn}) ---", flush=True)
        _fr, _info = C.run_episode(model, data, None, ctrl, ik, _rng, TASK,
                                   mode="act", verbose=True, on_frame=_watch)
        _ok, _why = hold_succeeded(model, data, _info)
        _tot.append(len(_fr))
        _tilt.append(_peak[0])
        print(f"  {len(_fr)} frames ({len(_fr)/C.FPS:.1f}s) | peak tilt "
              f"{_peak[0]:.1f} deg | {_ok} {_why}", flush=True)
    print(f"[cfg] median {sorted(_tot)[len(_tot)//2]} frames "
          f"({sorted(_tot)[len(_tot)//2]/C.FPS:.1f}s), "
          f"peak tilt max {max(_tilt):.1f} deg", flush=True)
    raise SystemExit(0)

def main():
    writer = imageio.get_writer(args.out, fps=C.FPS, codec="libx264", quality=8,
                                macro_block_size=1)
    banked = 0
    try:
        # Episodes 1 + 2 are run as a unit on one seed, and retried as a unit, so
        # a bad seed can never leave half a pair in the video.
        seed = args.seed
        pair_ok = False
        for _ in range(20):
            A = run_episode(EPISODES[0]["ep"], EPISODES[0]["tags"], seed)
            B = run_episode(EPISODES[1]["ep"], EPISODES[1]["tags"], seed)
            if A[1] and B[1] and A[3] == B[3]:
                pair_ok = True
                break
            print(f"  pair seed {seed}: A={A[1]} ({A[2]}) B={B[1]} ({B[2]}) "
                  f"fingerprint={'match' if A[3] == B[3] else 'MISMATCH'}"
                  " -- retrying", flush=True)
            seed += 1
        if not pair_ok:
            raise SystemExit("could not bank a clean matched pair")
        verdict = ("fingerprint A = B verified" if A[3] == B[3]
                   else "FINGERPRINT MISMATCH")
        print(f"matched pair on seed {seed}: {verdict} ({A[3][:16]}...)", flush=True)

        results = [(EPISODES[0], A), (EPISODES[1], B)]
        seed += 100
        for spec in EPISODES[2:]:
            for _ in range(20):
                r = run_episode(spec["ep"], spec["tags"], seed)
                seed += 1
                if r[1]:
                    results.append((spec, r))
                    break
                print(f"  ep {spec['idx']} seed {seed-1}: {r[2]} -- retrying",
                      flush=True)
            else:
                print(f"  ep {spec['idx']}: no clean episode found", flush=True)

        for spec, (frames, ok, why, fp, px) in results:
            e, tags = spec["ep"], spec["tags"]
            instr = TASK.format(colour=e["named_colour"], shape=e["named_shape"])
            if e["is_pick"]:
                head = f"ep {spec['idx']}/5  ·  ACT — pick up and hold"
            elif e["mode"] == "refuse_hazard":
                head = f"ep {spec['idx']}/5  ·  REFUSE — placard on the named object"
            else:
                head = f"ep {spec['idx']}/5  ·  REFUSE — named object not present"
            l2 = f'"{instr}"   ·   {spec["relation"]}   ·   {placard_text(tags)}'
            l3 = f"placard {px} px at decision frame"
            if spec["label"]:
                l3 += f"   ·   {spec['label']}"
            if spec["idx"] in (1, 2):
                l3 = f"matched pair {spec['label'][-1]}  ·  {verdict}  ·  " + l3
            accent = ACT_C if e["is_pick"] else REF_C
            lines = [(head, _FONT), (l2, _FONT_S), (l3, _FONT_S)]
            for f in frames:
                writer.append_data(annotate(f, lines, accent))
            banked += 1
            print(f"ep {spec['idx']}: {e['mode']:18s} {spec['relation']:6s} "
                  f"{placard_text(tags):34s} | {len(frames):4d} fr | "
                  f"placard {px:5d} px at decision | {why}", flush=True)
    finally:
        writer.close()
    print(f"\nwrote {args.out}  ({banked} episodes)")



if __name__ == "__main__":
    main()
