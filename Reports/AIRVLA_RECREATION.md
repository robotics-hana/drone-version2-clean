# AirVLA Recreation — Experiment Setup & Running Log

Recreating **"π, But Make It Fly: Physics-Guided Transfer of VLA Models to
Aerial Manipulation"** (Tucker et al., arXiv 2603.25038, airvla.github.io) on
this repo's SkyGrip platform, **in simulation only**, targeting eventual
deployment on the real robot. Started 2026-08-18. This file is the living
record: setup, decisions, deviations, limitations. Update it with every
change.

---

## 1. The paper in brief (verified against arXiv HTML v1, §refs)

- **Platform**: ModalAI Starling 2 Max + VOXL 2, PX4. **No arm** — a fixed,
  3D-printed UMI-style gripper (2 hobby servos) slung under the drone (§III-B1).
- **Policy**: public **π₀ base checkpoint**, fine-tuned (no LoRA/freezing
  mentioned anywhere → full fine-tune implied, unconfirmed) (§VII-A1/A2).
- **Observations**: three RGB cameras at 256×256, 5 Hz — one **external
  third-person**, one **onboard forward**, one **onboard downward**; plus
  mocap pose and gripper aperture (§III-B2).
- **Actions**: chunks A∈R^{H×D} of **4-DoF EE delta poses (x,y,z,yaw) +
  gripper**, padded to π₀'s native dims; "first 7 action dimensions"
  extracted at inference, **gripper is dim 7** (§III-B2, §VII-A2). 10 Hz,
  realized as PX4 position setpoints. Delta reference frame/convention:
  **not stated** (recreation decision D3 below).
- **Training**: chunk H=50, 30k AdamW steps, global batch 32, 1k-step warmup
  → peak LR 2.5e-5 → cosine to 2.5e-6 (§VII-A2).
- **Data**: 270 human teleop demos total ≈ 10 h (~120 manipulation, ~150
  navigation); navigation augmented to 200 trajectories with ~50
  Gaussian-splat synthetic "corrective" episodes (perturbed starts,
  gate-extremity recovery waypoints, terminal hover height U[1.0,1.5] m,
  post-gate waypoint ball r=0.125 m). Manipulation = teleop only (§III-E, §VII-A2).
- **Inference**: Real-Time Chunking (RTC), H=25, 10 denoising steps,
  exponential prefix-attention schedule; plus **Payload-Aware Guidance** —
  an inference-time term added to the flow-matching sampler:
  v_guid = v_θ − s(τ)·ξ, ξ = (∇_{x_τ}Â_θ)ᵀ∇_A Φ(Â_θ;o), with
  Φ_payload = (λ_z/2)·α·Σ_t w_t (z_t − z_des)², z_des = z_curr + Δz,
  α∈[0,1] from gripper close-intent over last K commands fused with measured
  aperture. λ_z=0.5, Δz=0.15 m, w_t=(t/(H−1))^γ, γ=1, K=4 (§III-D, §VII-A3).
  **s(τ) is never specified** — must be tuned (decision D6).
  NOTE: w_t here (payload, increasing toward late steps) ≠ RTC's w_t mask
  (decreasing); two different symbols-in-collision.
- **Tasks & eval** (§IV): 20 trials/task/method; staged success, later stages
  conditioned on earlier.
  1. *Penguin Grasp* — "pick up the stuffed animal and put it in the blue
     bin" (stages: pick, place). Object position varied within a box.
  2. *Gate Navigation* — "fly through the gate and hover over the stuffed
     animal" (stages: gate, hover). Gate trained/tested at left and right.
  3. *Compositional* — concatenated prompt, **unseen during training**
     (stages: gate, hover, pick, place; wrong-order = failure).
  - Headline numbers: pick/place — naive π₀ 50/0, +RTC 85/23.5, +guidance
    **100/50**; navigation with synthetic data 95/100; compositional 62.5%.
    ACT and Diffusion Policy (LeRobot defaults) ≈ 0 everywhere.
  - OOD: novel objects (chips bag 10/0, toy sandwich 70/57, box 30/33);
    novel gate positions (front 0, left 0, right 40).

## 2. Platform mapping (paper → ours)

| Paper | Ours | Note |
|---|---|---|
| Starling 2 Max, no arm, underslung gripper | SkyGrip drone + 2-DoF arm **frozen at the level stow pose** + Pololu gripper | v7 validated translate-and-grab; arm = rigid mount |
| PX4 position setpoints | PD flight controller position setpoints | same abstraction; our PD ≈ their PX4 |
| Mocap pose | Sim ground-truth pose | exact analog |
| Gripper aperture (servo) | `right_clamp` joint reading | direct analog |
| Teleop demos | **Scripted expert (FSM + PD), sim only** | core deviation; see Limitations L1 |
| Gaussian-splat synthetic nav data | Scripted "corrective" nav episodes in sim (same randomisations: perturbed starts, gate-extremity waypoints) | we already simulate; splat step unnecessary |
| Lab: mocap arena, table, blue bin, gate | Scanned lab scene (`SkyGrip_lab.xml`), blue mat, bin (to be painted blue), **gate to be added** | |
| Stuffed penguin | Rigid stand-ins: YCB mustard bottle + extended object set | MuJoCo can't do plush; see D2 |

## 3. Observation & action spec (ours)

**Cameras** (256×256 logged, 10 Hz — see D1):
- `camera1` = **wrist_cam as the downward view** (Hana's design, 2026-08-18):
  the arm holds the perpendicular grasp pose, so the wrist camera looks
  straight down over the jaws — the paper's downward onboard camera *with
  the gripper in frame*, exactly as theirs is (they had to composite
  gripper patches into synthetic downward views for this reason; ours has
  the jaws natively). No new camera added.
- `camera2` = **scene_cam** (forward onboard — unchanged)
- `camera3` = **external_cam** (static third-person in the lab — the
  world-fixed overview camera fills this role)

**Proprio state**: `[x, y, z, qw, qx, qy, qz, gripper_aperture, joint1,
joint2]` — pose + aperture mirror the paper; **arm joints are our addition**
(user requirement; constant during episodes but informative for real deploy).

**Action (7-D, π₀ single-arm EE-delta layout — decision D3)**:
`[Δx, Δy, Δz, Δroll≡0, Δpitch≡0, Δyaw, gripper]` at 10 Hz; deltas are
per-step, world frame. Gripper is dim 7 per the paper. Roll/pitch identically
zero (underactuated platform cannot command them independently) — matches
the paper's "4-DoF + gripper padded to 7".

## 4. Tasks (ours), prompts verbatim from the paper pattern

1. **Grasp**: "pick up the {object} and put it in the blue bin" — object on
  the blue mat, position randomised within a box; bin fixed. Stages:
  *pick* (object lifted ≥ 80 mm and held), *place* (object at rest inside
  bin footprint). Our quantitative operationalisation — the paper's is
  qualitative (Limitations L4).
2. **Gate Navigation**: "fly through the gate and hover over the {object}" —
  gate at left/right positions. Stages: *gate* (body crosses gate plane
  inside the aperture), *hover* (body within 0.25 m of over-object for 3 s).
3. **Compositional**: concatenated prompt, held out of training. Stages:
  gate, hover, pick, place; grasp-before-gate = failure.

## 5. Data plan (mirroring paper counts)

| Split | Paper | Ours |
|---|---|---|
| Manipulation demos | ~120 teleop | 120 scripted-expert episodes, sim |
| Navigation demos | ~150 teleop | 150 scripted nav episodes |
| Corrective nav | +50 splat-synthetic | +50 scripted corrective (same randomisation scheme) |
| Compositional | none (held out) | none (held out) |

Randomisations: object position box on the mat; object identity (see D2);
gate left/right; start pose jitter; lighting. Episode logging at 10 fps
(= action rate; D1).

## 6. Training plan

π₀ base via LeRobot 0.6 on sparks (pi0 policy code verified importable).
30k steps, global batch 32 (gradient accumulation as needed on the GB10),
AdamW, 1k warmup → 2.5e-5 → cosine 2.5e-6, chunk H=50. Full fine-tune to
match the paper's implied recipe; **fallback to LoRA only if the GB10
cannot hold full-FT memory — recorded as a deviation if taken** (risk R1).

## 7. Inference & evaluation plan

- Sim eval harness drives the trained policy closed-loop at 10 Hz through
  the same PD stack used for collection.
- **Methods, mirroring the paper's ablation ladder**: (a) naive π₀ (chunk
  H=50, replan at chunk boundaries), (b) π₀ + RTC (H=25, 10 steps,
  exponential prefix schedule), (c) π₀ + RTC + Payload-Aware Guidance
  (λ_z=0.5, Δz=0.15 m, γ=1, K=4; s(τ) tuned — D6; VJP via
  reconstruction-guidance approximation ∇Â≈I unless exact backprop proves
  necessary — D7).
- 20 trials/task/method; staged success conditioned as in the paper; OOD
  round: novel objects + novel gate position.
- ACT + Diffusion Policy baselines (LeRobot defaults) — optional phase 2.

## 8. Decision log

- **D1 (obs rate)**: paper images at 5 Hz (hardware limit), actions 10 Hz.
  We log everything at 10 Hz — a superset; noted as deviation.
- **D2 (objects, revised per Hana 2026-08-18)**: the penguin was incidental
  to the paper — no plush stand-in needed. Primary: YCB mustard bottle.
  Generalisation set: YCB meshes re-screened **with scaling** into the
  32 mm jaw envelope + the validated primitive vocabulary
  (blocks/cylinders/prisms). OOD holdouts mirror the paper's
  hard/medium/easy spread.
- **D3 (delta convention)**: per-step world-frame deltas,
  7-D layout `[Δxyz, Δrpy(rp=0), grip]`. Paper leaves this unstated.
- **D4 (arm, revised again per Hana 2026-08-18)**: the arm ARTICULATES —
  holding it rigid would make Link_1/Link_2 redundant. Behaviour: a
  camera-down TRAVEL pose while hovering/flying (wrist camera = the
  downward view), active EXTENSION to the perpendicular grasp pose to
  execute the pick, RETRACTION back to travel for the carry. The motion is
  platform-owned (a deterministic function of task phase — the paper's
  embodiment analog; its gripper is fixed, ours is an automated
  arm+gripper subsystem), never policy-commanded; the policy observes the
  joints in proprio and keeps the paper's exact 7-D action interface.
  Recorded alternative (not taken): 9-D actions giving the policy the arm
  — larger deviation from the paper.
- **D5 (payload physics) — RESOLVED 2026-08-18 by force-step probe**:
  weight-step on the gripper body while hovering under the PD. Peak /
  standing sag: 30 g → 20 / 2 mm (recovers 3 s); 100 g → 65 / 7 mm
  (recovers 6 s); **200 g → 148 / 147.5 mm, never recovers** (the PD's
  integrator only engages under 0.15 m error, so the sag is permanent);
  400 g → 298 mm standing. The paper's phenomenon exists on our platform
  at ≥200 g — and its guidance offset Δz=0.15 m matches our 200 g standing
  sag (0.147 m) almost exactly. **Primary experiment object: ~200 g bottle
  variant** (1.96 N weight vs ~5 N pinch — 2.5× grip margin); the 30 g
  bottle stays as a variation, where guidance should show ≈no effect (a
  useful built-in negative control).
- **D6 (s(τ))**: guidance schedule unspecified in the paper; we tune it and
  report the value used.
- **D8 (bottle-grasp geometry, measured 2026-08-18)**: four findings from
  the payload-sag probe's failure chain, all inherited by the collector:
  (a) the blue mat is visual-only — the bottle rests on the physics floor,
  cap centre z=0.183; (b) the pedestal "grip 13 mm below top" rule leaves
  2 mm of pad-to-neck margin on the bottle — aim the pinch at
  cap_top − 4 mm instead (11 mm margin); (c) GRASP_Y_OFFSET is a block-era
  housing clearance — on a ROUND cap it shifts the pinch off the diameter
  and causes melon-seed ejection; round objects get axis-centred pinches;
  (d) the 23 mm cap in the 32 mm jaws leaves 4.5 mm/side — a ~6 mm PD
  feed-forward residual is fatal, so the final descent uses the measured
  jaw-residual waypoint correction + a close gate, exactly the production
  run_episode discipline.
- **D7 (guidance Jacobian)**: paper defines a VJP but not its
  implementation; we start with the standard reconstruction-guidance
  approximation.

- **D9 (heavy payloads, 2026-08-18, filmed)**: cap-grasping the tall bottle
  at ≥100 g is a TIPPING-LEVER problem — ~0.35 N at cap height tips a 200 g
  bottle (half the force sliding it needs), the tilting cap drags the
  friction-locked jaws, and no controller fix exists because the cap is the
  only jaw-compatible feature. Falsified along the way (do not re-try):
  close-rate reduction, closing-hold brace, mirrored-actuation alone,
  mass-compensated contact solref, lift-through-grasp, grip-triggered
  ascent, low floor friction. Two changes were kept as faithful
  improvements: the mirrored left-jaw actuator and a 1.8 N gripper force
  limit. The 200 g payload moves to a purpose-built CALIBRATION-WEIGHT
  object (fat base + 16 mm grip knob, mass near the pinch) — to be authored
  as a real body in objects_lab.xml (runtime geom-morphing produced
  artifact-suspect results and was abandoned).
- **D10 (grasp-pose regression, 2026-08-18)**: `solve_drop(0.19)` now yields
  a 24°-tilted jaw mouth — it optimises CoM straightness, and the mass
  recalibration (Link_1 ×3.3) moved the optimum. Blocks tolerate the tilt
  (v5–v7 all validated post-recalibration); small round features do not.
  A 9°-tilt pose exists in the IK grid (select by `_mouthz` at the drop).
  Fix the shared IK when the weight object lands.

- **D11 (real gripper specs + collector reset, 2026-08-18, from Hana)**:
  measured hardware — grip strength **0.2–0.5 N**, max object **100 g**.
  These cohere (0.5 N × μ1.5 × 2 faces ≈ 1.5 N ≈ 100 g with margin) and are
  now the sim's gripper forcerange (±0.5 N). Arm servo limits rescaled ×3.3
  to the real arm mass (Hana's diagnosis: joints saturated 10–18% of every
  carry at the old limits, jerking carried objects loose). Bottle-cap
  carries also exhibited RATCHET CREEP (filmed clean mid-carry release with
  100× static margin) — partial pad engagement on a smooth round cap plus
  pendulum oscillation. Consequences: experiment payload = 100 g (transient
  sag 65 mm per D5 probe); primary manipulation object = the purpose-built
  `pick_weight` (fat base + 16 mm stem, full pad engagement, CoM at the
  pinch); the bottle stays as a light secondary object. Collector offsets
  to be re-derived from a droop-vs-height calibration rather than stacked
  runtime corrections (the stacked corrections were measured fighting each
  other after each platform-parameter change).

- **D12 (manipulation resolution, 2026-08-18/19)**: the working grasp chain,
  end to end — settle-with-hold discipline (corrections on arrival-tick
  transients were measured diverging), staged descent, settled 3-axis
  jaw-residual corrections (sub-mm), alignment-gated close, **pre-narrow to
  24 mm** (a full-open close ghosts through the 16 mm stem in the L7
  contact regime; pre-narrowed closes seat at ~7.6 mm aperture), grasp weld
  engaged at the clean pre-close moment (post-seat gating measures solver
  shove, not the grasp; weld eq_data layout unit-tested: anchor@0,
  relpos@3, relquat@6), gentle 0.12 m/s carry, release over the bin.
  **Success = placed** (an object cannot arrive at rest in the bin without
  a successful pick and carry; the mid-episode `held` sample misread
  weld-carried lifts and discarded physically perfect episodes).
  Validation: manipulation 3/3 banked in 3 attempts; navigation 7/7.

- **D13 (15-episode review feedback, 2026-08-18, from Hana)**: five items
  from the review of `airvla_review15.mp4`, resolved as follows.
  1. *"What is camera1?"* — the **wrist camera** (`wrist_cam`, logged as
     `observation.images.camera1`): the paper's downward, gripper-in-frame
     onboard stream, riding the arm per the embodiment design.
  2. *Wrist camera identity* — the `wrist_cam` XML definition had drifted
     during the session (a U20CAM-matching experiment left it at fovy 130,
     aimed nearly level). **Restored to the session-start view**
     (`pos="0 -0.04 0.082"`, fovy 110, angled down over the jaws) on user
     request; the verbatim session-start `xyaxes` rendered upside down in
     the restructured core's gripper frame (rotors at frame bottom), so
     both image axes are negated — same view axis, image upright.
  3. *External view missed half the lab* — the shared `overview_cam` was
     zoomed for the pedestal scene. Added a **lab-specific `lab_external`
     camera** (gate_lab.xml, high corner at (3.4, −2.4, 3.2), fovy 55)
     framing the whole blue mat, gate, bin and drone; `camera3` now maps
     to it. Verified by render.
  4. *Frame size / pixelation* — 256×256 is the paper's policy input
     resolution and is kept for the dataset; logging is now
     **2× supersampled** (rendered at 512, LANCZOS-downscaled to 256) for
     anti-aliased frames at identical dataset size, and review videos are
     rendered at 2× tile size for viewing.
  5. *Episode-1 dip ("falls down then gets back up")* — that is D5's
     payload-transfer transient caught on camera: the 100 g load lands on
     the airframe at lift-off and the PD sags ~65 mm before its integrator
     recovers, enough to set the weight back down. Fixed pilot-style, in
     the expert only (the platform must stay naive so Payload-Aware
     Guidance keeps its target at eval): **two-stage lift** — rise 12 cm,
     dwell 3 s while the sag is absorbed near the floor, then climb.
  Two incidents from implementing these fixes, both instructive:
  - A jaw-orientation re-selection of the arm poses (attempted for item 2
    before the true cause was found) was **reverted after a traced
    failure**: changing `q_grasp` changes the pad–stem contact geometry
    the close/seat physics was validated on — the traced episode ghosted
    its close (0.3 mm aperture), took a 0.34 m airframe shove at seat, and
    was then pinned at 0.15 m altitude by weld-vs-floor solver drag.
    `solve_drop(0.16)`/`(0.19)` stand. Do not re-tune grasp-arm poses
    without re-validating the close.
  - Restoring the camera by **scp'ing the whole locally-mirrored
    SkyGrip_core.xml silently reverted the pad contact fix** (friction
    1.5 → stale 3.0, solref 0.004 → stale 0.02): every close ghosted to
    ~0 mm aperture at solref 0.02's ~7 N/m fingertip stiffness — the
    documented pre-weld failure mode, reintroduced. Diagnosed by lining
    up the validated run's log signature (identical `pre_close`, ap
    7.6 mm) against the failing builds and auditing the cluster's
    `git diff` of the core. Pad lines restored; smoke re-validated 2/2 in
    2 attempts at ap 7.6 mm. **Rule: never push a whole mirrored file to
    the cluster to change one line — edit surgically, and audit
    `git diff` on the cluster after any model-file push.**

- **D14 (final grasp/carry protocol, forward reach, camera ownership,
  2026-08-19)**:
  - **Cameras are user-owned**: Hana hand-tuned `wrist_cam` and
    `scene_cam` in SkyGrip_core.xml — do not modify either. (The
    wrist-camera confusion traced to the arm-pose experiments flipping
    the world in a camera that rides the arm; the camera XML itself was
    innocent. A temporary red marker geom located the mount on the
    forearm near the elbow; removed once placement was settled.)
    `lab_external` (camera3) widened fovy 55 → 75: the whole lab is in
    frame (verified by render).
  - **Forward reach** (user request: "reaching backward … looks weird"):
    arm poses are the mirrored solve_drop solutions (−j1, −j2) looked up
    in the FK grid — the arm is not symmetric, so mirrored drops differ
    (travel 0.184 m, grasp 0.139 m) and offsets are taken from the grid,
    never sign-flipped.
  - **Final weld/close protocol** — five variants falsified by trace
    before the working one:
    1. no weld during seat → free stem ghosts through pads (ap 0.0);
    2. fixed pre-close datum → weld-vs-floor fight, ~24 s integrator
       stall, slingshot +0.86 m / −0.26 m ("falls down then gets back
       up");
    3. single post-seat datum re-capture → wind-up already accumulated,
       slingshot +0.99 m / −0.40 m;
    4. continuous 0.2 s re-capture → pads ratchet through the stem
       (ap 0.0);
    5. any ground-level seat dwell at the FORWARD pose → transfer-sag
       weld stretch on the long forward lever CAPSIZES the airframe
       (thrust projection collapsed to 0.9 N, drone on its side —
       pd_flight thrust = a_des·R z-column).
    **Working protocol**: weld at pre-close (gated on <8 mm measured
    alignment) → close to the STEM WIDTH (grip 0.45, near-zero commanded
    intrusion — full close was driving 8 mm/side of ghost penetration,
    the root shove) → immediate datum re-capture + pre-close integrator
    restore → NO ground dwell → two-stage lift (+0.12 m, tol 0.10, 3 s
    dwell absorbs the D5 transfer sag ~10 cm up) → carry.
  - **Payload-referenced arrival**: under load the PD trims to a steady
    offset from its waypoint (measured 14 cm behind, 7 cm low; the
    position integrator is gated off until near-target), so any
    body-position release tolerance either deadlocks or drops off-bin.
    The expert now arrives coarse (tol 0.18), then corrects its goal by
    the *weight's* measured residual over the bin centre and releases
    when the payload is within 8 cm — the settled-residual discipline
    from the jaw corrections, applied at the drop.
  - **Verified episode** (seed 71000): pick/place/success all true, seat
    aperture 8.8 mm (pads resting on the stem), worst carry sag **5 mm**
    (was 383 mm), no slingshot (peak 0.676 m), release dead-centre
    (1.401, 1.599) vs bin (1.4, 1.6). The user-visible dip is gone; the
    steady loaded droop remains, deliberately — it is the honest naive
    platform behaviour that Payload-Aware Guidance targets at eval.

- **D15 (yaw channel live + dead-hover removal, 2026-08-19, from Hana)**:
  two review items on the same flight leg.
  - *"Turn after pickup so the bin is actually in the scene camera"* —
    the **dyaw action channel is now live** (supersedes the earlier
    no-yaw decision, and matches the paper's 4-DoF x,y,z,yaw action
    space): after lift-off the expert yaws the nose (scene camera, body
    −y axis; yaw = atan2(tx, −ty) for a world bearing) onto the bin and
    flies *forward*. Nav episodes now **spawn facing their direction of
    flight** (yaw π), so the gate is in the forward camera throughout —
    as in the paper. The PD's yaw error is wrap-aware (pd_flight.py), so
    holding π is safe. Verified by mid-carry renders: the bin is centred
    in camera2 through the transport and drop.
  - *"Drone hovers for a prolonged time after pickup"* — the loaded
    cruise droop sits above any tight settle tolerance, so the
    settle-based climb stages burned their full timeouts (~21 s of dead
    hover, measured). Final sequence (amended per review, "bring the
    drone to a hover then turn"): rise 12 cm until the payload is
    measurably airborne → 2.5 s sag dwell → climb to hover altitude with
    an altitude-threshold + near-zero-vertical-speed check (a position
    tolerance would deadlock on the droop) → 1.5 s stable hover → yaw
    onto the bin → fly forward. Turning from stable hover also cut the
    weight-carry sag to 5 mm (was 27 mm turning right off the pickup);
    the bottle shows ~60 mm of pendulum bob during the turn (0.19 m
    object swinging on the cap pinch), which is motion of the payload,
    not thrust sag.

- **D16 (object variety + paper-task audit, 2026-08-19, from Hana)**:
  - *Variety*: manip/nav episodes now alternate between **two task
    objects** — the 100 g calibration weight (stem pinch, close 0.45,
    seat ~8.8 mm) and the **empty 30 g mustard bottle** (cap pinch per
    D8, close 0.70, seat ~11.2 mm on the 23 mm cap). Prompts are
    templated ("pick up the {obj} …", "… hover over the {obj}"); the
    non-task object parks in-scene as background clutter; the grasp weld
    retargets per episode via `m.eq_obj2id`. Two masses also give the
    payload-guidance experiment two sag regimes. The object set is
    everything the hardware can handle: prepare_objects.py screened 22
    YCB candidates and only the mustard bottle is both graspable (23 mm
    cap vs 32 mm jaws) and stable upright; heavier objects are excluded
    by the 100 g hardware ceiling (D9/D11). Verified: both objects
    pick-and-place cleanly (bottle release at (1.405, 1.586)).
  - *Paper tasks audit*: the paper trains **two** tasks — object
    pick-and-place ("Penguin Grasp" → our weight/bottle-to-bin) and
    **Gate Navigation** (→ our gate task, LEFT/RIGHT positions, plus
    corrective near-miss recoveries). Its third evaluation, the
    **compositional prompt** ("fly through the gate and put the {obj} in
    the box"), is *held out from training* and used only at eval — we
    reproduce that design: no compositional episodes are collected, and
    the prompt is reserved for Phase E.

- **D17 (approach/carry visibility fixes, 2026-08-19, from Hana)**:
  - *Bottle knocked over on approach* — two stacked causes, both fixed:
    (1) the travel→grasp arm swing is a 37° arc on the mirrored elbow
    branches, and swinging while descending diagonally swept the pads
    through cap height (the 0.19 m bottle is in the arc; the 0.06 m
    weight never was) — the swing now happens **at altitude** (+0.30 m)
    followed by a **vertical** descent; (2) the 24 mm pre-narrow leaves
    only 0.5 mm/side over the 23 mm cap, less than the 4.5 mm alignment
    gate, so an off-centre pre-narrow shoved the bottle (measured
    −9.9 mm pre-close, object knocked 0.14 m) — pre-narrow and gate are
    now **per-object** (weight 24 mm/4.5 mm; bottle 27 mm/2.0 mm).
  - *"Object floating in space" in the forward camera* — the scene
    camera cannot see the under-body region where the grasp poses hang.
    After the hover-and-turn, the arm extends to a **forward carry
    pose** (shallowest-drop, furthest-forward grid pose): gripper and
    welded object sit mid-frame in camera2 for the whole transport; the
    pd_flight arm-CoM feed-forward absorbs the shifted trim.
  - *Cameras* — wrist_cam fovy briefly set to 130 to match scene_cam
    (user message), then **reverted to 110 on user decision**; the
    hand-fixed mount/orientation was never altered. Both cameras remain
    user-owned. A stale TEMP-marker comment was removed from the core
    (the marker geoms themselves were already gone; the mocap marker
    spheres are alpha-0 and cannot appear in frames).
  - *Arm-swing timing* — the object briefly read as "floating" in the
    forward camera because the climb/hover/turn happened while the arm
    was still slewing (~10 s at the safety-limited arm rate) from grasp
    to carry pose, leaving the payload in the camera's under-body blind
    spot. The expert now extends the arm immediately after lift-off and
    **waits at 12 cm until the arm arrives** before climbing — the
    payload stays just off the mat (shadow context, reads as "just
    picked") and the whole climb/turn/carry has the gripper visibly
    holding the object. Residual: ~8 s of arm-out-of-frame during the
    low reconfiguration; eliminating it entirely would mean grasping at
    the forward pose, which requires re-validating the close physics
    (D14 rule).

- **D18 (bin approach: "drove over and past the box", 2026-08-19, from
  Hana)**: three stacked causes, each measured and fixed —
  1. *payload lead*: at the forward carry pose the object leads the body
     by the arm's reach, so a body-aimed goal flies the payload past the
     centre → transport goal pre-compensated by the live-measured hang;
  2. *arrival glide*: the loaded lateral PD is soft, and carried cruise
     momentum glided the body ~0.6 m past an abruptly-stopped setpoint →
     two-leg approach: carry-speed to a waypoint 0.35 m short, **brake
     until ground speed < 3 cm/s**, then creep the final leg at 0.05 m/s
     (SP_STEP_FINAL);
  3. *static loaded lead*: the payload's pitch moment on the forward arm
     is unmodelled by the platform's arm-CoM feed-forward, so the body
     PARKS ~0.28 m ahead of its setpoint at 100 g (~0.08 m at 30 g) —
     the dominant term → the final-leg goal is shortened by the
     **live-measured lead** (body-vs-setpoint offset sampled at the
     brake hover).
  Result: pre-release overshoot 345 mm → **41 mm** (weight), 0 mm
  (bottle); both payloads released centred. The release itself remains
  payload-referenced (D14). Committed as V0.39.0 (local 5c00f16, cluster
  fbd194a); review videos >99 MB are gitignored per GitHub's limit.

- **D19 (grasp chain without the pre-grip shuffle + sequence reorder,
  2026-08-19, from Hana)**: review asked for (a) sequence = grab → hover
  up → extend arm → turn → approach → drop, (b) a snappier drop, (c) no
  "reorientation thing" before the grip (the forward-back nudging that
  knocked the mustard). Removing the at-depth nudges initially collapsed
  bottle yield to ~25%; four measured iterations rebuilt it *without*
  reintroducing the shuffle:
  1. the loaded FK/droop **z-bias (~5 mm low)** — previously absorbed by
     the at-depth nudges — put the pads at the cap's tapered bottom edge
     where the close levers the bottle over; now corrected in the
     pre-descent all-axis pass, +4 cm above the object where the open
     pads are clear of everything;
  2. a creep-speed descent was tried and reverted: 5 s of loaded y-drift
     produced bimodal +11 mm landings (cap pressed into the mouth's back
     wall); the descent runs at carry speed (0.5 s, no drift time);
  3. the old gate → pre-narrow → pause → close order left ~4 s of hover
     drift between the alignment check and the weld capture; reordered
     to **pre-narrow → gate → weld+close back-to-back**;
  4. the alignment gate is now **anisotropic**: tight on the closing
     axis x (graze mechanism; lands sub-mm), tolerant along the mouth
     depth y (±6.5 mm just seats the object deeper/shallower).
  Verified: bottle 4/4, weight 2/2, zero pre-grip nudging in any of
  them. Drop timing tightened (final creep 0.08 m/s, shorter settles,
  0.6 s fall wait).

- **D20 (tucked cruise + extend-at-the-box, drop latency, scene tidy,
  2026-08-19, from Hana)**:
  - *Arm extends only at the box* (user proposal, endorsed after
    analysis): the cruise flies with the arm at the grasp pose — the
    payload pitch-moment lead (~0.28 m at 100 g) that the extended arm
    imposed on the whole transport largely disappears — and the ~10 s
    arm swing overlaps the approach flight (body-aimed first leg stops
    0.45 m short; the brake waits for stillness AND arm arrival; hang
    and lead are both measured fresh at the brake; the creep leg aims
    the payload dead-centre). Trade-off accepted in review: the object
    is not visible in the forward camera during the cruise (the box is,
    dead ahead); it rises into frame with the arm at the drop.
  - *Drop latency*: measured 18–35 s above-box-to-release on weight
    episodes. Root cause: the post-creep settles measured body-vs-goal,
    but with lead compensation the body parks `lead` away from the goal
    BY DESIGN, so those settles always burned their full timeouts.
    Replaced with sp-arrival + stillness waits; the payload-residual
    check is the real arrival criterion. Now ~5 s typical (occasional
    ~20 s when a correction round fires).
  - *Scene*: the race gate is parked 3 m underground during
    pick-and-place episodes (visible only in nav); the mustard bottle is
    **60 g (half-full)** — the empty 30 g bottle kept getting knocked
    over; doubling the mass doubled tip resistance (5/5 grasp yield on
    verification) and keeps a sag regime distinct from the 100 g weight.

- **D21 (fallen-bottle recovery grasp, 2026-08-19, from Hana)**: "when
  the mustard falls, reorientate the gripper [and drone] to the new
  position of the cap and pick it up." Implemented as live-pose grasp
  targeting throughout:
  - `live_target()` computes the aim from the object's CURRENT pose; for
    a fallen bottle (axis z < 0.7) it returns the lying cap centre
    (+6 mm outward along the axis so the jaw mouth clears the shoulder)
    and a yaw that puts the closing axis perpendicular to the bottle —
    the drone and gripper reorient to wherever the object actually is.
  - The whole grasp is a two-pass loop: any missed close reopens,
    ascends clear, and re-aims at the live pose (a knocked bottle gets
    chased once). The close itself is **gated on the verified pre-close
    pose** — a blind close on a missed lying cap was measured rolling
    the bottle up to a metre.
  - Corrective slots now alternate **fallen-bottle recovery** (bottle
    spawns knocked over, settled to rest before the episode starts) and
    nav-corrective — the paper's corrective-demo concept applied to
    manipulation.
  - Gate axes were rotated into the **gripper frame** (after a
    reorienting yaw, world-x gating let 6.5 mm of closing-axis error
    through the 2 mm gate).
  - Known residual: ~40% of fallen attempts graze the free-rolling
    cylinder during the descent (2–4 mm per-seed drift vs 4.5 mm
    clearance) and self-discard; banked data therefore contains only
    clean recoveries, at ~1.7 collection attempts per corrective slot.

- **D22 (flight stability at the box, 2026-08-19, from Hana)**: "the
  flight looks unstable … go to the side or edge, stay at hover, then
  extend arm"; "the arm and drone are overshooting by the box." Two
  interacting causes: (1) extending the arm **mid-flight** shifts the
  CoM while translating and the attitude loop visibly chases it;
  (2) even extending at hover, the unmodelled arm+payload moment drifts
  the body ~0.3 m forward *during* the extension — done at the bin edge,
  that parked the drone over the box and forced a visible back-up.
  Final sequence: cruise tucked → hover **0.75 m short** of the bin
  (drift happens in open air) → extend the arm at dead hover →
  re-stabilise → measure hang + lead → one monotonic forward creep to
  the drop point. Verified: **0 mm overshoot on both objects**, worst
  carry tilt ~7° (normal flight lean), drops centred.

- **D23 (cap tracking site + aperture-confirmed pickup, 2026-08-19,
  from Hana)**: two refinements to the fallen-bottle recovery.
  - **`mustard_cap_site`** (objects_lab.xml): an invisible group-4 site
    at the cap centre — MuJoCo's FK provides the exact live cap pose in
    any orientation, and the recovery aims at it (offset 6 mm outward
    along the bottle axis so the jaw's inner edge clears the fatter
    neck). Data-collection scaffolding only; alpha-0 + group 4 means it
    cannot appear in a training frame.
  - **Aperture-confirmed pickup**: for the fallen grasp, "when the
    clamp no longer shuts, we have the mustard" — close on any
    plausible pose; if the clamp physically stops at cap width
    (~11 mm), the cap is between the pads → weld at the seated pose and
    lift; a full shut means a miss → reopen and re-aim at the live
    site. The upright grasps keep the validated pre-close-gated
    protocol (D14). Verified: fallen 3/5 (aperture reads 11.2 mm on
    every success, misses self-identify at full-shut and self-discard),
    upright bottle 4/4, weight 2/2. The mustard's random fall
    directions are covered by the perpendicular-yaw reorientation
    computed from the live site pose.

- **D24 (anti-cheat weld guards + the honest knock rate, 2026-08-19,
  from Hana: "episode 4 the mustard fell and the gripper still picked
  it up as if it was upright?????")**:
  - The weld could catch a bottle **mid-tip** at a transiently-aligned
    instant and carry it frozen at a tilt — diagnostics looked nominal
    (pre-close 3 mm, aperture 11.2) because they measure jaw-vs-target
    geometry, not the object's state. Guards added: the upright-path
    weld also requires the object **at rest and standing** (R22 > 0.95)
    at the weld instant, plus a post-close **aperture sanity range**
    per object (weight 5–12 mm, bottle 8–14.5 mm) — outside it, the
    weld is released and the pass retries.
  - Banning cheaty welds exposed the true upright-bottle knock rate:
    ~50% of approaches disturb the bottle (all earlier "5/5" runs were
    partly cheat-inflated). Root mechanism: the tight settles **time
    out under loaded trim** (their False return accumulated into `ok`
    but gated nothing), so descents launched tens of mm off target and
    swept the open jaws through cap height. Descents are now
    **hard-gated on live jaw alignment** (<10 mm to descend, abort
    upward at >25 mm mid-descent), grasp passes retry up to 3× with an
    at-rest wait before re-aiming, and episodes bank only clean grasps.
    Honest yields: upright bottle ~50% per attempt (acceptance loop
    absorbs it, ~2 attempts/episode), weight 100%, fallen recovery
    ~60–100%. Efficiency-only cost; flagged for a dedicated tuning
    session if Phase C collection time matters.
  - Scene/UX in the same round: **wooden box** (procedural grain
    texture wood.png; prompt updated to "put it in the wooden box"),
    and **stationary-hover bookends** (1 s at episode start and end) —
    clean rest-state boundaries for chunked-action training, matching
    real deployment.

- **D25 (domain variety + real-drone colours, 2026-08-19, from Hana)**:
  anti-overfit randomisation across every episode —
  - **Start positions widened for all tasks**: manip x ±0.5 / y 0.9–1.6
    / z 0.55–0.95; nav lateral ±0.35 around the gate, y −1.9..−1.2,
    z 0.60–1.00.
  - **Target box moves and re-tints per episode** (position x 1.1–1.7,
    y 1.2–2.0; wood-tone tint over the grain texture), and a **second
    distractor box** (bin2, its own random grey-ish tint, random far
    placement) is always in scene — the prompt's referent stays unique
    ("the wooden box" = the grain-textured one). All manip navigation,
    turn, approach and success logic now uses the per-episode bin pose.
  - **Very-top cap pinch**: AIM_Z = cap top − 4 mm (was −7 mm; the deep
    pinch was a ratchet-era fix the weld obsoletes) — "the rest of the
    bottle is too wide."
  - **Real-drone colours**: black-plastic body, arm, gripper and legs;
    **blue propeller discs** (visual-only geoms over the mesh rotors).
    Verified by render and by full-pipeline runs (successes land in
    their per-episode randomized bins; yields at the honest D24 rates).

- **D26 (release aesthetics + final variety + clean banking,
  2026-08-19, from Hana)**:
  - *"Looks like the drone is throwing the object"* — correct diagnosis
    by review: it is the **unload pop**, the counterpart of the D5
    pickup sag (the PD's payload thrust trim releases the instant the
    weld opens and the drone jumps upward as the object falls), plus
    ~0.4 m of drop height. The expert now **descends and releases with
    the payload ~10 cm above the rim** (verified release altitudes
    0.32 m weight / 0.48 m bottle, everything landing upright); the
    honest unload pop itself is kept — naive-platform behaviour the
    guidance experiment needs.
  - *Forward camera*: scene_cam XML is byte-identical to the user's
    hand-fix — the apparent change was the recoloured body and new blue
    prop discs entering its wide 130° view; the discs were shrunk
    (r 0.095 → 0.082) and raised out of the view edge.
  - *More variety*: task-object spawns widened (x ±0.6, y 0.2–1.0) and
    the parked distractor object jitters ±0.25 m per episode (drone
    start + box position already randomize per D25).
  - *Clean banking* (episode 8: an 1,825-frame triple-retry marathon
    with a gate-failed 6.6 mm-off pinch banked because it eventually
    placed): episodes now bank only if the final grasp **passed the
    alignment gate** and the episode is **≤1,100 frames** — marginal
    cocked pinches and marathon slogs self-discard.

- **D27 (grasp dwell + scene_cam nose mount + release height,
  2026-08-19, from Hana)**:
  - **0.3 s grasp-confirmation dwell** between close and lift-off — as
    on the real platform, the pinch is confirmed before committing.
  - **Release height raised to rim + 22 cm** (was rim + 10 cm): the
    unload wobble happens directly over the box, and the extra margin
    keeps the drone clear of the walls while it restabilises — still
    far below the carry-height release that read as "throwing".
  - **scene_cam raised to a nose mount** (user direction: "a bit higher
    and the propellers not in view"): pos moved from (0, −0.115, 0.03)
    — which sat between the front rotors, letting them intrude into the
    130° view — to (0, −0.20, 0.055), ahead of the prop discs' front
    edge. Orientation and fovy remain the user's hand-tuned values.
    Render-verified clean at hover and at 8° cruise tilt (only the
    physical rotor shadows remain in frame). Also verified this round:
    camera2 ≡ scene_cam mapping byte-identical on both machines.

- **D28 (fallen-recovery honesty chain, 2026-08-19, from Hana: ep15
  "grasping air ... should rotate and look down and pinch from
  above")**: four stacked findings —
  1. *Tilted-settle routing*: a bottle propped at an angle (axis z
     0.7–0.95) fell into the UPRIGHT approach; any non-upright bottle
     (axis z < 0.95) now takes the reoriented overhead grasp.
  2. *Air-welds*: the fallen aperture check was **circular** — the
     close commands cap width (11.2 mm), so an air-close also stops at
     11.2 and "confirms" — and the 15 mm plausibility ball let the weld
     carry the bottle floating beside the jaws. Now: anisotropic
     gripper-frame plausibility (6 mm across the closing axis, 15 mm
     along the cap where deep/shallow seats are harmless) plus a
     **seated-verify** (live cap site within 13 mm of the jaws after
     the close) before any weld; not-seated closes reopen and retry.
  3. *See-through geoms audit* (user request): collision-vs-visual
     overlay rendered on a lying bottle; vertex-computed alignment
     showed the visual mesh only 0.8–1.5 mm off the collision axis —
     corrected to the exact computed offset (0.0145, 0.0220, 0.0032).
  4. *The bottle rolled forever*: MuJoCo's default condim 3 has **no
     rolling friction**, so a knocked-over cylinder never stops on the
     flat mat — every recovery pass aimed at a stale pose (traced: the
     bottle's heading drifted continuously all episode). Object geoms
     now use **condim 6** with rolling friction 0.008, which the
     already-present coefficients needed to act at all.
  Result: only genuine seated recoveries can bank (fallen ~40% per
  attempt, self-discarding; upright bottle and weight unaffected).

## 9. Limitations (running)

- **L1 — no teleoperation**: all demos are scripted experts in sim. Expert
  trajectories are cleaner and less multimodal than human teleop; this may
  make imitation easier and RTC/guidance deltas smaller than the paper's.
- **L2 — no plush objects**: MuJoCo deformables are impractical here; rigid
  stand-ins change contact character vs a penguin.
- **L3 — our gripper's envelope is narrow** (32 mm jaws): object diversity
  requires mesh scaling; native-scale generalisation is limited.
- **L4 — success criteria operationalised by us**: the paper's stage
  definitions are qualitative; our numeric thresholds are stated in §4.
- **L5 — visual domain**: our external camera sees a scan-reconstructed lab
  (its own artefacts documented in PIPELINE_EDIT_LOG §8b area); the paper's
  cameras see the real world. For eventual real deployment, the scan gives
  a head start but is not photoreal at close range.
- **L6 — π₀ checkpoint identity**: "the public π₀ base checkpoint" is not
  pinned to an artifact in the paper; we use LeRobot's π₀ base port and
  record its hash.

## 10. Phase plan (review gates in bold)

- **A. Scene & task build**: down_cam, blue bin, gate model, object set
  screen, 10 Hz logging path. → payload-sag probe (D5).
- **B. Collector**: `collect_airvla.py` — three task programs, delta-action
  logging, staged-success checkers. → **15-episode annotated sample +
  3-camera video for Hana's review. STOP for approval.**
- C. Full collection (120 + 150 + 50), dataset audit (visibility, action
  stats).
- D. π₀ fine-tune (30k steps) + training-curve report.
- E. Eval harness: naive / RTC / RTC+guidance, 20 trials × 3 tasks + OOD;
  results tables mirroring the paper's.
- F. Real-robot transfer prep (out of scope until E reads well).

## 11. Camera/task visibility tracking

Per the standing requirement: every dataset build gets a segmentation-based
visibility audit (target px per camera per phase — the v5/v6 audit tooling)
plus a task-legibility check on the review video (object identifiable at
the decision point; command on screen). Results recorded here per build.

*(build entries will be appended below as they happen)*
