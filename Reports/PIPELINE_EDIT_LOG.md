# SkyGrip pipeline edit log — 2026-08-15/16

Record of every change made during the recalibration → dataset → v5 sessions,
on both machines. "Local" = this repo (`drone-version2`, Windows). "Cluster" =
`sparks:~/Drone-Arm_VLA` (DGX Spark; all cluster changes are **uncommitted**
working-tree edits, recoverable via git).

---

## 1. Mass recalibration to the real hardware (2026-08-15)

Real weighings supplied by Hana: **drone 736 g, arm (excl. gripper) 331 g,
gripper 30 g → all-up 1097 g.** Both models previously had body 904 g (from a
superseded 1179 g all-up weighing) and drastically too-light arms.

| File | Change |
|---|---|
| Local `Sim'n'Real/Mujoco/SkyGrip_core.xml` | base_link 904→736 g (inertia ×0.8142); Link_1 95→314.45 g, connect 5→16.55 g (arm ×3.31, split kept 95:5); gripper already 30 g. CoM positions kept; inertia scaled with mass. |
| Cluster `SkyGrip_full.xml` (old monolith) | Same targets: base ×0.8142; Link_1 95→184.97 g + connect 75→146.03 g (×1.9471); old gripper 85+10+10→30 g (×2/7). Later superseded by the model port (§4). |

**Open assumption:** the Link_1:connect split is proportional; confirm where the
elbow servo actually sits (it sets the arm CoM → roll coupling).

Validation: PD flew both recalibrated models at the historical success rates;
MPPI comparison videos in `Reports/mppi_pick_trial_cluster*.mp4` (0/2 both
physics; one 180° flip on recalibrated — heavier arm, same MPPI precision floor).

## 2. Grasp gate + re-grasp retry (cluster `collect_demos.py`, then local)

Motivation: 90% baseline (18/20) — both failures were closes at the 12 mm
tolerance boundary (jaw on block edge → push-out, or close too high → drop).

- **Close gate:** GRASP's close only starts once the live jaw→object lateral
  error < `CLOSE_GATE_TOL = 9 mm` **and** the jaws are not more than 8 mm above
  the commanded aim (z judged vs `servo_to`, so retries may deliberately aim high).
- **Re-grasp retry:** after LIFT/HOLD_OBJECT (cluster) or LIFT (local, V0.18
  place-task variant), if the object did not rise, re-plan DESCEND→GRASP→LIFT
  from the object's **live** position plus the measured residual, up to 2 tries.
  Downward chase clamped to 15 mm (body must stay clear of the crash margin);
  objects that left the table (>60 mm drop or off surface) are not chased.
- Local V0.18 `collect_demos.py` got the same gate+retry adapted to its
  PLACE/RELEASE pipeline.

Measured: cluster 90% → **96.7%** (58/60, seeds 11–13). Local 91.7% (55/60) —
remaining local failures are place-phase misses, not grasp failures.

## 3. Dataset conventions (cluster)

- v3 15-ep and v4 15-ep PD datasets: `hanapasta/pick_refuse_hazard_v3_pd15`,
  `..._v4_pd15` (v4 cut: seed 61000 + `--limit_slots 11` → exactly 15 episodes:
  4 atomic pairs, 4 ungrounded, 2 act_free, 1 act_tag).
- `CAMERAS` dict (both repos): added `"camera3": "overview_cam"` (SmolVLA's full
  named camera set).

## 4. New-airframe port to the cluster (2026-08-16)

The cluster still simulated the old drone/arm/gripper. Ported this repo's
current model:

- Copied to cluster: `SkyGrip_core.xml` + `pedestal_task.xml` +
  `mocap_markers.xml` + meshes (`base_link.STL`, `Link_1.STL`, `connectf.stl`,
  `gripper_base.stl`, `right_clamp.stl`, `left_clamp.stl`).
- Cluster's V0.37 code needs named scene geoms + hazard tags that this repo's
  `pedestal_task.xml` lacks → extracted the V0.37 scene verbatim from the old
  monolith into **`pedestal_task_v37.xml`** (cluster-only file).
- Cluster `SkyGrip_full.xml` is now a 3-include composition:
  `core + pedestal_task_v37 + mocap_markers`.
- Backups on cluster: `SkyGrip_full_oldgripper.xml`, `base_link_old.STL`,
  `Link_1_old.STL`.
- Cluster code constants adapted to the Pololu gripper: `GRIPPER_OPEN`
  0.017→**0.016** (actuator's full 32 mm range), grasp aim depth
  `half_h − 0.010`→**`− 0.013`** (10 mm clear column above grasp_site → 3 mm
  margin; validated V0.19 envelope).
- Datasets v3/v4 regenerated on the new airframe (visual spot-check of decoded
  frames confirmed new claws + new body in all cameras).

## 5. Camera3 zoom + video quality (2026-08-16)

- `SkyGrip_core.xml` (both machines): `overview_cam` fovy **70 → 40**. At 70 the
  20 mm objects rendered ~10 px and colours did not survive compression;
  at 40 the drone + table stay framed and colours are discernible
  (verified by render at the v5 fixed table height).
- v5 datasets encode at **crf 20** (v3/v4 used LeRobot's default 30) — done in
  `collect_v5.py` by wrapping `pick_rgb_encoder`.

## 6. V5 pipeline (cluster: `collect_demos.py` hooks + new `collect_v5.py`)

Design goal: keep every v4 **rule** (atomic A/B pairs, hazard tags + bystanders,
fixed-length INSPECT decision dwell with act/refuse parity, trapezoidal ramped
DESCEND, sidecars, gate metrics, slot schedule) but replace the **approach**
with a fixed clean corridor, for SmolVLA + π-VLA training.

`collect_demos.py` additions (all inert unless the v5 flags are set — v3/v4
behaviour is bit-identical when `V5_OBSERVE is None`):

- `FIXED_PED_HALF_H` — fixes the work surface height; the random draw is still
  consumed so the RNG stream stays aligned with v3/v4 on identical seeds.
- `V5_OBSERVE = dict(back, up, start_up)` + `_opening_lead()` — replaces
  HOVER_START with: **SETTLE** (1.2 s calm hover, far back + high) →
  **OBS_DESCEND** (straight down) → **OBSERVE** (2.0 s fixed dwell; scene_cam
  sees the objects' tops AND side profiles — this is where colour/shape ground
  the instruction) → **ADVANCE** (single linear forward+up leg to over the
  named object). SETTLE/OBSERVE run their full budget (added to the
  never-"arrive" dwell set); new `PHASE_FRAME_BUDGET` entries.

`collect_v5.py` (new, cluster): reuses `collect_v4.main()` wholesale with:

- `spawn_v4` overridden → fixed yaw-0 corridor spawn at the SETTLE point
  (no azimuth sectors, no FOV gate — geometry fixed and verified by render).
- `FIXED_PED_HALF_H = 0.20` → surface always 0.40 m.
- `V5_OBSERVE = dict(back=0.35, up=0.20, start_up=0.55)` — probed by render:
  ~29° elevation at ~0.40 m; tops + sides + colours readable.
- Central-span placement: `PEDESTAL_X_LIMIT` 0.26→**0.20**, `OBJ_MARGIN`
  0.035→**0.045**. Pulls the *allowed placement zone* inward so all on-table
  objects are well-framed at OBSERVE; object-to-object separations (band
  distances, `BYSTANDER_MIN_SEP`) unchanged.
- crf 20 encoding (§5).

Episode-length note: SmolVLA/π-VLAs train on sampled action chunks, so episode
length is free; the SETTLE/OBSERVE/HOLD dwells give smooth starts and ends
(~450–670 frames vs v4's ~440–550).

Dataset: `hanapasta/pick_refuse_hazard_v5_pd15` (15 episodes, same seed-61000
slot cut as v4_pd15). Final collection 2026-08-16: 15 episodes / 8,698 frames,
100% slot-attempt success, 142 MB (crf 20 + longer episodes vs v4's ~46 MB).

## 7. Visibility audit + ground-tether probe (2026-08-16)

- `v5_visibility_check.py` / `v6_visibility_check.py` (cluster): replay
  episodes through the collection path, segmentation-render camera1/camera2
  every 3rd frame, count target pixels per phase.
- **v5 findings:** opening fully sighted (both cameras, every frame), but
  EXTEND is a ~1 s both-camera blind window, GRASP is occluded at contact, and
  scene_cam never sees the object after ADVANCE (inherent to overhead grasps).
- **`tether_probe.py` / `tether_band.py` (cluster): the documented "forward
  grasp flips the drone" failure DOES NOT REPRODUCE on the recalibrated
  airframe with the Pololu gripper** (old result: snap-shut gripper, wrong
  masses). Close + lift at full forward reach: < 2° attitude, clean 410 mm
  lift. Workable band: `reach_y ∈ [−0.175, −0.155]` (both object heights);
  shallower pushes the block off the table. This supersedes the old warning in
  the build_plan comments.

## 8. V6 pipeline — side grasp at forward reach (2026-08-16)

Hana's design: approach from the side, table in view, arm extending toward
parallel with the table, grasp at reach. Implemented as v5 + a new tail:

- `collect_demos.py`: `V6_REACH_GRASP` flag; act plans ADVANCE to a hover
  BEHIND the object row, then REACH_OUT → INSPECT (decision dwell at reach) →
  DESCEND → GRASP → LIFT → HOLD_OBJECT, all at the reach pose — the body never
  flies over the object. Refusals fly the identical reach opening and diverge
  after INSPECT (v4 contrast preserved). Close-gate + re-grasp retry unchanged.
- `collect_v6.py` (new, cluster): imports the v5 configuration, then
  `REACH_FRACTION = 1.0`, `REACH_Y_RANGE = (−0.175, −0.155)` (the validated
  band), `V6_REACH_GRASP = dict(rise=0.14)`, REACH_OUT budget 16 s (the swing
  runs at the gentle 0.10 rad/s REACH_SLEW and uses most of it).
- **v6 audit: zero both-camera-blind frames in any phase**; scene_cam keeps
  the target in view through approach, decision, close (~500 px) and hold
  (~400+ px). Episodes lengthen to ~800–1,020 frames.
- Dataset: `hanapasta/pick_refuse_hazard_v6_pd15` (same seed-61000 slot cut).
  Final collection: 15 episodes / 13,315 frames, 100% slot-attempt success,
  223 MB.

## 8b. Forward-level wrist at stow (2026-08-18, cluster V0.38.1)

Hana spotted in the v6 review video that the stowed forearm (Joint_2) tips
the wrist camera upward. Measured: the legacy `solve_drop(0.13)` stow is
(3.1°, −27.5°) → wrist view pitch **+14° above the horizon** (sky). New
`TRAVEL_POSE = (0.160, −0.320)` keeps a comparable tuck (drop 0.156 m,
lateral −0.050) with the wrist camera level and forward (−1.2°), making
camera1 a second forward view of the scene through SETTLE/OBSERVE/ADVANCE.
Default `None` keeps v3/v4 bit-identical; enabled in collect_v5 (v6
inherits). v6_pd15 dataset + review video regenerated with the fix.

## 8c. Design decision: no yaw action channel (2026-08-18)

Considered and rejected for v5/v6: yaw never varies in the fixed-corridor
demonstrations, so a yaw channel would be a constant output (zero learning
signal, one new rollout failure mode — spurious yaw at forward reach slings
the arm CoM about the vertical axis). Heading stays platform-owned, like
thrust/attitude; the policy observes it via the state quaternion. Revisit
only alongside demonstrations that exercise it (multi-azimuth scenes via the
v4 sector machinery, or real-hardware deployment) — never bolt a constant
channel onto corridor data. Action space stays 6-D across v3-v6.

## 8d. V7 glide-capture pipeline (2026-08-18, cluster V0.38.2)

Hana asked why the drone reaches down at all instead of keeping the stow
configuration (claws parallel to the table) and flying straight into the
grasp. Answer: it was impossible under the old 0.20 m legs — but the V0.18
short legs (0.15 m) changed the arithmetic, and probes confirmed the
horizontal capture works: **±5–8 mm lateral, ≥ ±10 mm vertical window, zero
object drift during the glide, attitude ≤ 3°** (`horizontal_probe.py`,
`vertical_probe.py`).

V7 (`collect_v7.py`, `V7_GLIDE` flag): SETTLE → OBS_DESCEND → INSPECT
(decision dwell at the vantage; hazard placard measured 492–602 px) → DROP
to grip height → level forward glide → GRASP → LIFT → HOLD_OBJECT. Refusals
share the opening and retreat after INSPECT. The arm NEVER articulates.

Rationale locked to the research goal: the refusal-direction ablation study
needs act-vs-refuse to diverge along one legible axis so post-ablation
behaviour reads as a 1-D spectrum — pure compliance (act-indistinguishable
glide+grasp on the forbidden object) vs physical reaction (stall, slowed
approach, close-without-approach, oscillation). Trade-off accepted: joint
channels are constants in v7 demos; v6 remains the articulated variant.

Dataset: `hanapasta/pick_refuse_hazard_v7_pd15` (same seed-61000 cut).

**Capture fix (V0.38.3):** the first full run banked 13/15 at 53% slot
success. Deterministic repro of the worst slot (red cylinder, 6 straight
failures) found three stacked defects: a level glide cannot capture a
cylinder at all (18–22 mm diameter vs the mouth's ~10 mm clear depth —
bulldozed 52 mm; the probes had only tested boxes); hop=25 mm left the pad
bottoms grazing tall-object tops once DROP's transit tolerance was spent;
and GLIDE's transit-tolerance handoff gave the descent 14–16 mm of lateral
error, landing a pad on the cylinder's edge. Final geometry: glide at
**45 mm above the grip point**, jaw-servo'd GLIDE arrival (12 mm), short
vertical finish. Repro slot now succeeds; dataset regenerated.

## 9. VLA precision-gap investigation: grasp assist (2026-08-16)

Question: can a chunked, ~cm-accurate policy do a ~10 mm grasp on this
platform? Files (cluster): `grasp_assist.py` (the platform layer),
`assist_probe.py` (noise-injection experiment: per-chunk-constant setpoint
bias sigma, scheduled close, 8 eps/cell), `assist_debug.py` (traces).

**Baseline confirms the diagnosis**: sigma=10-20 mm chunk bias -> close-onset
errors 12-44 mm vs the 9 mm window -> 25%/25%/12%/0% success at
sigma=10/15/20/30. This is the observed VLA training-failure mechanism.

**Iterated design, each step falsified by measurement** (all attempts kept in
the file's docstring): integral trim (winds up chasing chunk bias — WORSE
than baseline, mirrors the collector's own disabled servo); proportional trim
added to the policy setpoint (wrong equilibrium — cancels exactly half the
bias); measured-jaw servoing (self-excites via attitude sway on the 0.16 m
reach lever); align-at-grip-height (open jaws touch and shove the block,
corrupting the aim).

**Final design** (in `grasp_assist.py`): engage only on the policy's own
close intent within a 40 mm xy basin (half the min object separation — can
never capture a neighbour); a near-miss interlock (WAIT, 1.2 s, no aiming);
align laterally at +30 mm only when low-and-misaligned, else descend; static
waypoints from level-attitude FK (never servo on measured jaws); release the
close through a 9 mm / z-window gate; hold the frozen waypoint while the pads
seal; hand back with the residual frozen through a 0.5 s reference low-pass.

**Result**: seal alignment SOLVED — pad-seal error 11 mm mean (max ~20-35) vs
35-62 mm baseline; success 5/8, 5/8, 2/8, 0/8 vs 2/8, 2/8, 1/8, 0/8.
**Residual failure mode is marginal pinch** (10-15 mm blind-axis seals hold
only sometimes), the same physics the collector's RE-GRASP retry eliminates
(90%→96.7%). Conclusion: assist at rollout + a grasp-failure signal
(gripper aperture stalling near zero = closed on air) driving a policy- or
harness-level retry is the complete answer; alignment alone is not.

## Where things live

- Cluster datasets: `~/.cache/huggingface/lerobot/hanapasta/` (none pushed to
  the HF Hub).
- Local copies: `C:\Users\hanah\.cache\huggingface\lerobot\hanapasta\`.
- MPPI/PD comparison videos: `Reports/mppi_pick_trial*.mp4`.
- All cluster code/model changes are uncommitted in `~/Drone-Arm_VLA`.
