# Chapter 3 — System and Experimental Setup (draft, ~5,500 words)

> DRAFT for the dissertation. Numbers are taken from the project's live
> record (Reports/AIRVLA_RECREATION.md, decisions D1–D36) and reflect the
> system as actually run for the final dataset (V0.42.0) and training run.
> Rewrite in your own voice; citation keys are placeholders.

## 3.1 Overview and design principles

This chapter describes the experimental infrastructure built to recreate
the physics-guided VLA transfer method of Tucker et al. [ref: arXiv
2603.25038] on a different aerial platform, entirely in simulation. The
recreation target is the paper's full pipeline: demonstration collection
on an aerial manipulator, fine-tuning of the public π₀ base
vision-language-action model on that data, and closed-loop evaluation with
the paper's inference-time ablation ladder (naive chunking, Real-Time
Chunking, and Payload-Aware Guidance).

Three principles governed every design decision, and are worth stating
before the details because they explain choices that would otherwise look
arbitrary.

**Fidelity before convenience.** Where the paper specifies a quantity —
action space, chunk horizon, optimizer schedule, dataset composition — we
adopt it unchanged. Where the paper is silent (the delta reference frame,
the success thresholds, the fine-tuning regime), we make an explicit
recorded decision and mark it as ours. A running decision log (36 entries
at the time of writing) records every such choice with its rationale; the
load-bearing entries are summarised throughout this chapter and the full
log is reproduced in Appendix [X].

**Physical honesty.** Simulated grasping can "succeed" for reasons that
have no physical counterpart: contact-solver artifacts can carry an object
that was never actually pinched, and a scripted expert can exploit
privileged state in ways no real system could. Because the dataset is
intended to train a policy for eventual real-world deployment, the
demonstration system was engineered so that every banked grasp is one the
real hardware could plausibly execute, enforced by measurement-based gates
described in §3.3.4. Five alternative grasp-approach designs were built
and falsified with measurements during development; they are reported with
the results because rejected designs are evidence, not waste.

**Provenance.** Every episode in the final dataset carries the exact code
version, random seed, and scene state that produced it, making the
dataset reproducible end-to-end from a single command.

## 3.2 Simulation platform

### 3.2.1 Vehicle, arm and gripper

The platform is *SkyGrip*, a quadrotor (736 g) carrying a 2-DoF
underslung arm (331 g) terminating in a parallel-jaw gripper (30 g,
32 mm maximum aperture), modelled in MuJoCo [ref]. This differs from the
paper's platform — a ModalAI Starling 2 Max with a *fixed* UMI-style
gripper and no arm — and the mapping between the two embodiments is a
central design question of the recreation.

The resolution (decision D4) is that the arm *articulates* but is
**platform-owned**: its pose is a deterministic function of task phase — a
camera-down travel pose during flight, an extension to the perpendicular
grasp pose for the pick, and a forward carry pose while loaded — and is
never commanded by the policy. The policy observes the arm joints in
proprioception but retains the paper's exact 7-dimensional action
interface (§3.4.2). The arm thus plays the role that the fixed gripper
mount plays on the paper's platform: part of the embodiment, not part of
the action space. The alternative — a 9-D action space giving the policy
the arm — was recorded and rejected as a larger deviation from the paper.

The flight stack is a PD position controller accepting position and yaw
setpoints at the same abstraction level as the paper's PX4 setpoint
interface, with an arm-aware centre-of-mass feed-forward. A payload
force-step probe established the platform's sag characteristics (100 g
payload: 65 mm transient, 7 mm standing offset, ~3 s recovery), which are
the physical phenomena the paper's Payload-Aware Guidance term exists to
manage at inference time.

### 3.2.2 Laboratory scene and real-world registration

The simulated environment is a photogrammetric reconstruction of the real
laboratory (163k triangles, 82 MB of texture), giving all three cameras
visually realistic backgrounds rather than a sterile void. The working
surface is a raised 40 mm blue mat matching the real lab's three
6.00 × 1.52 × 0.04 m gym mats, so that the simulated floor maps 1:1 to
the real floor.

The scene is metrically registered to the physical laboratory: a Polycam
point-cloud scan of the real room (1.01 M points) was segmented and
aligned, verifying among other things that the scanned mat surface sits
at exactly +40 mm over the scanned floor — matching the simulated riser —
and producing a stored rigid transform between simulation coordinates and
the scan frame for later sim-to-real work. Task furniture comprises a
wooden target box, a deliberately light-and-cool-tinted distractor box
(dark tints proved confusable with dark wood in review), a race-style
gate (0.90 × 1.06 m opening) for the navigation task, and parked
non-task objects as background clutter.

The external camera position was not hand-tuned: it is placed 30 cm
inside the scanned room mesh's own wall corner nearest the target box
(measured from the mesh vertices), so the third-person viewpoint
corresponds to a physically mountable position in the real room.

### 3.2.3 Manipulands

The paper's primary manipuland is a stuffed penguin. Deformable-body
simulation at contact fidelity sufficient for grasping is impractical in
this setting (Limitation L2), so the recreation uses a rigid penguin
stand-in with deliberate mass distribution: a ballasted base (17 g), an
ellipsoid body, a graspable head sphere (8 g) proportioned to the
gripper's 32 mm envelope, and a solid, collidable beak. The ballast makes
the penguin tip-stable — it cannot fall over when nudged — while the
solid beak constrains the grasp: the gripper must approach with the
correct heading so the beak exits through the open jaw mouth rather than
being crushed (D32). The second task object is a 100 g cylindrical
weight, matching the payload class used in the paper's payload analysis.
All object geometry uses six-dimensional contacts with rolling friction;
the MuJoCo default three-dimensional contact has no rolling resistance,
and knocked cylinders rolled indefinitely until this was corrected.

### 3.2.4 Grasp transmission model

The most consequential fidelity problem in the platform is grasp
mechanics. The real gripper demonstrably carries a 100 g object at
0.2–0.5 N of pinch force; MuJoCo's contact solver, given the model's
3-gram fingertip bodies, cannot reproduce this — the solver either drops
the object or generates non-physical interpenetration forces. The
recreation therefore models a *successful pinch* as a weld constraint
between gripper and object — but, critically, the weld is only permitted
to engage from a **measured seated pinch**: the jaws must be within 8 mm
of the grasp target, the object upright and at rest, and — after the
close — the measured aperture must land inside the object's
characteristic seated band (9.0–13.5 mm for the penguin head, 5–12 mm
for the weight). A close that stops anywhere else (air, mid-tip, neck) is
released and retried. The weld thus encodes the real hardware's
demonstrated capability at the moment the simulation can verify the
geometry of an honest pinch, rather than papering over solver limits
unconditionally. The same measured-pinch gate is used at evaluation time
(§3.6), where it cannot fabricate success because task scoring inspects
the object's final pose.

## 3.3 Expert demonstration system

The paper's demonstrations are human teleoperation (270 episodes,
~10 hours). Ours are generated by a scripted expert — a finite-state
machine over the PD stack — which is the recreation's largest single
deviation (Limitation L1): scripted trajectories are cleaner and less
multimodal than human teleoperation, which may make imitation easier than
in the paper. The compensating benefit is scale, repeatability and zero
operator time; the final dataset was collected unattended overnight.

### 3.3.1 Tasks and prompts

Three task families mirror the paper's:

1. **Manipulation** — "pick up the {object} and put it in the wooden
   box": approach, grasp, lift, carry, and release into a box whose
   position, rotation and wood tint vary per episode.
2. **Gate navigation** — "fly through the gate and hover over the
   {object}": traverse a gate placed at left/right positions, then hover
   over the named object.
3. **Corrective** — recovery-flavoured episodes alternating between
   edge-of-workspace object spawns (recovery-to-coverage) and perturbed
   navigation starts, standing in for the paper's ~50 Gaussian-splat
   synthetic corrective episodes: since our environment is already
   simulated, the paper's splat-synthesis step is unnecessary and the same
   randomisation scheme is applied directly (Limitation L8).

The compositional task (concatenated prompt) is **held out of training**
exactly as in the paper, as is a paraphrased manipulation prompt ("put
the {object} in the wooden box") reserved for instruction-generalisation
evaluation.

### 3.3.2 Expert control architecture

The expert owns a ramped setpoint (position, yaw, grip fraction) slewed
toward a goal each 10 Hz tick; the recorded action is the per-tick delta
actually commanded, so the dataset's action stream is exactly the signal
that reproduces the trajectory when integrated. Yaw goals are remapped to
the nearest 2π-equivalent of the current setpoint before being adopted
(D36), so commanded turns always rotate the shorter direction — a
correction introduced after grasp-heading randomisation exposed
long-way-around turns (a full mid-carry spin) whenever the numeric gap
between two headings crossed the ±π seam.

### 3.3.3 The manipulation program

A manipulation episode proceeds: hover-in → nose-first approach → grasp →
two-stage lift → short-way turn → tucked cruise → staged release. The
elements that took the most engineering, and their reasons:

**Approach.** The drone turns nose-toward the object at a dead hover,
flies straight at it (placing the object in the forward camera's view
throughout the approach), brakes 0.55 m short, swings the arm to the
grasp pose at that standoff, then slides once to directly overhead. The
slide target rotates the arm's body-frame kinematic offset by the
commanded grasp yaw (D36) — subtracting the offset unrotated is correct
only at zero yaw, and at randomised grasp headings had parked the jaws
beside the object. For the penguin the grasp yaw is *heading-aware*: the
beak direction is read from the object's live pose and the drone yaws so
that the beak exits through the open jaw mouth and the pads pinch the
sides of the head.

**Descent and close.** All lateral alignment happens at altitude; the
descent is strictly vertical, hard-gated (it does not begin until the
jaws are within 10 mm of centred, and aborts upward if drift exceeds
25 mm mid-descent), because horizontal motion of open jaws at object
height was the empirically identified knock mechanism in earlier designs.
Alignment gates are evaluated in the *gripper frame* — 2.0 mm on the
closing axis, 6.5 mm along the jaw mouth — since after a reorienting yaw
the closing axis no longer aligns with any world axis. The close commands
the object's width (not zero), the weld engages per §3.2.4, a 0.3 s
confirmation dwell mirrors real-platform practice, and the lift is
two-staged (rise 12 cm, verify the object is airborne, dwell through the
payload sag transient) before the carry.

**Carry and release.** The drone turns the short way toward the box,
cruises with the arm tucked (extending mid-flight shifts the centre of
mass while translating), brakes short of the box, measures the payload's
actual hang offset at a dead hover, then creeps the final approach and
releases at a payload-referenced height above the rim, verifying the
payload residual before departing.

### 3.3.4 Integrity safeguards

Because a scripted expert has privileged access to simulator state, the
pipeline distinguishes sharply between *aiming* (the expert may read
object poses — a human teleoperator has eyes too) and *succeeding* (gated
only by measurements a real system could make). Episodes are banked only
if the final grasp passed the alignment gate, the post-close aperture is
in the seated band, and the episode completed within 1,100 frames
(marathon episodes with repeated retries self-discard). During
development this honesty was expensive — an earlier manipuland showed an
honest ~50% knock rate under a since-rejected approach design — and the
five falsified grasp designs (documented in Appendix [X]) were rejected
on measured failure modes: integral wind-up, half-bias equilibria,
servo self-excitation through the arm's reach lever, contact-corrupted
aiming at grip height, and a behind-and-creep approach that swept the
object during horizontal creep (4/6 success against the final protocol's
15/15).

### 3.3.5 Domain randomisation

Per episode: object spawn position (x ± 0.60 m, y 0.20–1.00 m) and
heading (uniform); drone start position (three-axis) with ±15° heading
jitter, decoupled from the gate position in navigation episodes so the
policy must find the gate rather than inherit alignment; target box
position (x 1.1–1.7 m, y 1.2–2.0 m), rotation (uniform yaw) and wood
tint; distractor box position and (light-cool) tint; parked-object jitter.
The placed-in-box success check evaluates the object inside the rotated
box frame.

## 3.4 Dataset and data pipeline

### 3.4.1 Format and composition

Data is stored in LeRobot dataset format [ref] and published to the
Hugging Face Hub. The final training dataset (`airvla_full`) comprises
**320 episodes — 120 manipulation, 150 navigation, 50 corrective —
mirroring the paper's ~120/150/+50 composition**, collected from a single
seed in one unattended run. Collection statistics are themselves evidence
of expert quality: **320/320 attempts banked with zero discards and zero
grasp retries**; post-close apertures were 10.9 ± 0.1 mm (penguin) and
10.4 mm (weight) across all manipulation episodes. A 15-episode sample
dataset with an annotated three-camera review video was regenerated after
every behavioural change and human-reviewed before the full collection
was authorised; the final sample was the fourth consecutive
zero-discard collection.

### 3.4.2 Observation and action spaces

**Cameras** (three RGB, 512×512, 10 Hz): `camera1` is the wrist camera
looking down over the jaws in the grasp pose — the paper's downward
onboard view *with the gripper natively in frame* (the paper had to
composite gripper patches into synthetic downward views; ours needs no
compositing); `camera2` is the nose-mounted forward camera; `camera3` is
the external third-person camera at the measured room-corner position.
Frames are stored at 512×512 native rather than the paper's 256×256:
π₀ consumes 224×224 regardless, so storage resolution is free at
collection time and future-proofs the dataset for higher-resolution
policies (D33). Images are logged at 10 Hz where the paper's cameras ran
at 5 Hz (hardware-limited); ours is a superset (D1).

**Proprioception** (10-D): position, orientation quaternion, gripper
aperture fraction, and the two arm joint angles — pose and aperture
mirror the paper; the arm joints are our embodiment's addition.

**Actions** (7-D at 10 Hz): per-step world-frame deltas
`[Δx, Δy, Δz, Δroll≡0, Δpitch≡0, Δyaw, grip]` (D3 — the paper does not
state its delta convention). Roll and pitch are identically zero: the
underactuated platform cannot command them independently, matching the
paper's "4-DoF + gripper padded to 7".

### 3.4.3 The scene-state sidecar and a feature-schema trap

Each frame additionally logs a 20-D `scene_state` vector (full poses of
both task objects, both boxes, and the gate) so that a later
Gaussian-splat re-render of the dataset can reproject every episode
without re-simulating physics. This exposed a subtle and instructive
pipeline trap: LeRobot's feature-mapping utility types **every**
`observation.*`-prefixed key as a policy state input, and π₀ pads state
to 32 dimensions — a naively named sidecar would have silently fed
ground-truth object poses into the policy, invalidating every result.
The sidecar is therefore deliberately stored under an unprefixed key,
which the feature mapper ignores; this was verified against the policy's
resolved input features before training.

### 3.4.4 Throughput

An unattended collection run proceeds at roughly wall-clock real time
(the physics steps far faster than real time; three-camera rendering and
H.264 encoding dominate): manipulation episodes ≈ 2 minutes each,
navigation ≈ 15 s of flight, giving the full 320-episode dataset in
~11 hours including hub upload — against the weeks of operator time the
equivalent teleoperation campaign would cost.

## 3.5 Policy and training configuration

### 3.5.1 Checkpoint and fidelity audit

The policy is the public π₀ base checkpoint via its LeRobot port
(`lerobot/pi0_base`; the similarly named `lerobot/pi0` repository carries
a stale configuration schema and does not load under LeRobot 0.6). The
paper nowhere mentions LoRA or parameter freezing, implying a full
fine-tune; the recreation makes this explicit and audited it against the
implementation's defaults: `freeze_vision_encoder=false` and
`train_expert_only=false` — i.e. the **entire trunk (PaliGemma VLM
backbone, vision encoder, and action expert) trains** — are pinned
explicitly on the training command line so the configuration is visible
in the run record rather than inherited silently.

### 3.5.2 Hyperparameters

Matching the paper's stated recipe (which the LeRobot π₀ defaults
reproduce): chunk horizon H = 50, 30,000 AdamW steps, 1,000-step warmup
to a peak learning rate of 2.5 × 10⁻⁵ with cosine decay to 2.5 × 10⁻⁶
over exactly 30k steps. Camera features are renamed at load time to π₀'s
native trio (external → base, wrist → left-wrist, forward → right-wrist
slots). Training runs in bfloat16 — π₀'s original training precision —
with gradient checkpointing.

**Deviation (batch size).** The paper trains at global batch 32. The
available machine (an NVIDIA GB10 with 121 GB of unified CPU/GPU memory,
shared with other users' jobs) cannot hold full-fine-tune state at that
batch: full-precision attempts were OOM-killed at batch 32, 16 and 8,
and bfloat16 with gradient checkpointing fits at **batch 8**, which is
the run configuration. This quarters the effective batch and is recorded
as the training-side deviation; the schedule was not rescaled. (A
LoRA fallback was held in reserve but never needed.)

### 3.5.3 Infrastructure notes

Two operational findings are recorded because they are reproducibility
hazards for anyone repeating the setup. First, on unified-memory hardware
the failure mode of an oversized run is not a Python CUDA-OOM exception
but a silent host-level OOM-kill during model loading; dataloader worker
processes contributed enough to the load-phase peak that the run only
loads reliably with zero workers. Second, checkpoints (full model +
optimizer state, ~8.3 GB) are written every 5,000 steps with a one-command
resume path, after a mid-run interruption demonstrated the cost of
their absence. Steady-state throughput is ~5–6 s/step, giving ~40 hours
for the 30k-step schedule on the shared machine.

## 3.6 Evaluation protocol (summary)

Evaluation details and results appear in Chapter [4]; the setup chapter
records the protocol design. The trained policy is evaluated closed-loop
in the same simulator through the same PD stack used for collection, at
10 Hz. The policy receives exactly the training observation set (three
cameras and proprioception) and the task prompt; it commands the same
7-D action interface. The platform-owned arm follows the same
deterministic automaton as in collection, driven only by commanded
signals (grip, altitude) and the weld state — never by object ground
truth — and the measured-pinch weld gate of §3.2.4 applies unchanged.
Ground truth is consulted only by the scoring functions, which evaluate
the paper's staged success criteria (pick/place; gate/hover; the
compositional chain with wrong-order failure) under our quantitative
thresholds (object lifted ≥ 80 mm and held; placed at rest inside the
rotated box footprint; gate plane crossed inside the aperture; hover
within threshold of the named object), the paper's own stage definitions
being qualitative (Limitation L4).

The method ladder mirrors the paper's ablations: (a) naive chunking
(H = 50, replan at chunk boundaries); (b) Real-Time Chunking (H = 25, 10
denoising steps, exponential prefix schedule); (c) RTC plus
Payload-Aware Guidance (λ_z = 0.5, Δz = 0.15 m, γ = 1, K = 4, with the
guidance scale s(τ) tuned, as the paper leaves it unspecified). Twenty
trials per task per method, with instruction generalisation probed by the
held-out paraphrase and compositional prompts, and an out-of-distribution
round with novel objects and gate positions.

## 3.7 Summary of deviations from the paper

| # | Paper | This work | Class |
|---|---|---|---|
| 1 | Human teleop demos (~10 h) | Scripted expert, simulation | Core (L1) |
| 2 | Real-world data & eval | Simulation throughout | Core |
| 3 | Plush manipuland | Ballasted rigid stand-in | Forced (L2) |
| 4 | Fixed underslung gripper | 2-DoF arm, platform-owned automaton | Embodiment (D4) |
| 5 | Splat-synthetic corrective nav | Scripted corrective (same randomisations) | Simplification (L8) |
| 6 | Images 256×256 @ 5 Hz | 512×512 @ 10 Hz (π₀ consumes 224) | Superset (D1, D33) |
| 7 | Global batch 32 | Batch 8 (bf16, grad ckpt; memory-bound) | Resource |
| 8 | Success stages qualitative | Quantitative thresholds, stated | Operationalisation (L4) |
| 9 | Grasp force via real gripper | Measured-pinch-gated weld model | Fidelity model (L7) |

Every deviation is either forced by the sim-only setting (1–3, 9),
an explicit embodiment mapping (4), a strict superset of the paper's
signal (6), or a stated resource constraint (7); none changes the
method under test — the π₀ transfer recipe and its inference-time
ablation ladder are implemented as published.
