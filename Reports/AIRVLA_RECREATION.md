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
