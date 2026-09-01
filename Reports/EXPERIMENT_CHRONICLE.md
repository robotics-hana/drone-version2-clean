# AirVLA recreation — the full pipeline, in order

This document walks the project from simulation platform to the v2 design,
in the order the work was actually carried out. Every Python file is named
where it enters the story, and every experiment is described in full
sentences: what it aimed to do, what we expected, and what the result
showed. Companions: `AIRVLA_RECREATION.md` (the decision log, D1–D50+),
`TEST_FILES.md` (the one-line catalogue with metric definitions),
`results_ledger.html` (the verified numbers), `eval_logs/` (raw outputs).

---

## Stage 0 — The simulation platform

**Files: `Sim'n'Real/Mujoco/SkyGrip_core.xml`, `SkyGrip_airvla.xml`,
`Sim'n'Real/Mujoco/collect_airvla.py`**

The SkyGrip drone (736 g) carries a two-joint arm (331 g) with a gripper
(30 g), modelled in MuJoCo with masses and inertias matched to the real
hardware. `collect_airvla.py` is the heart of the platform: it holds the
`Runner` (scene randomisation through `reset_scene` — table position,
object placement, bin positions, wood tints, object yaws, all drawn from a
seeded random stream), the scripted expert (a privileged-state pilot that
flies the task perfectly), and the PD flight controller with lead
compensation that converts high-level setpoints into stable flight. This
split matters architecturally: the learned policy will only ever command
7-dimensional world-frame setpoint deltas at 10 Hz; a classical controller
handles the fast, chaotic physics underneath. The task scene is a table
carrying two candidate objects side by side (a plush penguin and a
dumbbell weight), a wooden goal box, a distractor box, and — for
navigation episodes — a gate.

## Stage 1 — Dataset collection

**Files: `collect_airvla.py` (collection main + quality harness),
`collect_demos.py` (earlier demo tooling). Dataset:
`hanapasta/airvla_full` on HuggingFace.**

The scripted expert flew 400 episodes — 225 pick-and-place (the prompted
object must go into the wooden box) and 175 navigation (fly through the
gate and hover over the named object) — recorded at 10 Hz through three
cameras (a wrist camera, a nose camera, and a fixed external room camera)
plus a 10-dimensional state vector, with the scene's ground truth stored
alongside in a 20-dimensional `scene_state` sidecar. Every episode passed
an 18-gate quality harness before admission (takeoff clean, grasp
achieved, placement verified, no gate contact, and so on). Two design
facts recorded here became important much later: the two candidate
objects always spawn side by side at the same table depth, offset only
sideways, and which one is the target is *not* predictable from position
— only the language prompt carries it.

## Stage 2 — Training

**Infrastructure: LeRobot 0.6 fine-tuning of π₀ (4.03 B parameters) from
`lerobot/pi0_base`, MEAN_STD normalisation, state padded to 32
dimensions. Job scripts on the UCL Myriad cluster (SGE) and the Sparks
DGX (Slurm).**

The camera streams were mapped into π₀'s three input slots (the external
camera into `base_0_rgb`, wrist and nose into the two wrist slots), and
the model was fine-tuned at batch 4 on a single A100 at roughly 1.2
seconds per step. Checkpoints were taken at 30,000 steps, extended to
60,000, and later to 90,000 (1.34 passes over the data — trained, at that
point, with **no held-out episodes**, a limitation flagged on every
result it touches). A batch-8 variant at 30,000 steps was trained as a
control. All checkpoints are archived on HuggingFace
(`hanapasta/airvla_pi0_30k`, `_60k`, `_90k`, `_30k_b8`). A 40-episode
holdout list was defined later (`eval_logs/holdout_split.json`) and is
excluded from every training run from the camera-swap arm onward.

## Stage 3 — Evaluation, and the discovery that the harness was broken

**File: `Sim'n'Real/Mujoco/eval_pi0.py` (the working harness).**

The evaluation harness samples fresh scenes from a seeded stream, runs
the policy closed-loop, and scores each episode (closest approach in
millimetres as the primary metric; pick, place, and success flags as
secondary). The first full ladder on the 30k checkpoint scored **zero
successes in twenty episodes**, and instead of accepting that at face
value we audited the instrument. Two cheap tests were built to ask
whether the test rig itself could pass:

- **`harness_audit/open_loop_probe.py` (teacher-forced probe).** Aim:
  show the policy training images and compare its predicted actions
  against the expert's, with no simulator involved, to separate "the
  policy learned nothing" from "the harness blocks everything". We
  expected mediocre correlations if the policy was the problem. The
  result was per-channel correlations around 0.9 — the policy had
  clearly learned the task's structure, so suspicion moved to the rig.
- **`harness_audit/replay_diag2.py` (ground-truth replay).** Aim: feed
  the expert's own recorded actions through the evaluation rig on a
  byte-faithful copy of the recorded world; if the expert's own actions
  cannot pass, no policy can. The first version of this test was itself
  flawed (re-running scene reset re-drew random values and invalidated
  the replay), which was found and fixed by snapshotting the full
  simulator state at the first expert tick. The repaired replay
  **failed**, proving the harness could not have measured a successful
  pick from anyone.

The audit found five defects: a stale arm-deployment gate tuned to an
older scene geometry, a grasp weld that triggered on gripper aperture
alone and could tow objects through the air, an evaluation start
distribution that overlapped the training start distribution in zero of
225 episodes, an action clip slightly tighter than the expert's own step
size, and contact flicker on the plush object that defeated the weld
logic. **`harness_audit/fixfit5.py`** then fitted the repair —
demo-calibrated arm poses, a contact-debounced weld, corrected tuck
timing — with a pre-committed acceptance test: the ground-truth replay
must pass through the repaired rig, which it did, three of three. The
repairs ship as the `--platfix` and `--startfix` flags, and "honest
harness" everywhere in the project means those flags on.

## Stage 4 — Reproducibility: the frozen protocol

**Files: `eval_frozen_v1.py` (sha `a8cc240ba921`), `eval_frozen_v2.py`
(sha `1ec61ae930ca`).**

π₀ is a flow-matching policy whose sampling draws unseeded noise, so two
identical evaluations differ unless the torch generator is seeded; the
`--torchseed` flag fixed that, and every run since prints a `PROV` line
(script hash, checkpoint, full argument list, seeds, timestamp) before
any result. The evaluation script was then frozen: all comparable re-runs
use the frozen copy, never the evolving working file. This discipline
caught its first incident within a day — a flag added *after* the v1
freeze was silently ignored by a queued job, which the PROV line exposed;
the accidental run was kept as a legitimate extra baseline, the protocol
was re-frozen as v2, and the run was repeated correctly. Log filenames
and tags are never reused; superseded results are kept and marked.

## Stage 5 — The honest results ladder

**Runs through `eval_frozen_v1.py`; raw logs in `eval_logs/`.**

With the repaired, seeded, frozen harness: the 60k checkpoint navigates
well (12 of 20 honest navigation successes) and produced the project's
first genuine grasp. The citable pick baseline is the frozen n=60 run —
**median miss 242 mm, 3 of 60 picked up (5%, exact CI ≈ 1–14%), zero
placed** — with every episode on video. A later frame-by-frame review
(prompted by Hana watching the clips) sharpened the claim: the `picked`
flag is a latched criterion (weld engaged and object lifted 12 cm for at
least one tick), and of the three flagged episodes two are sustained
carries of roughly 40 and 58 seconds while one is a momentary
grab-and-drop, so the honest phrasing is "2 carries + 1 transient grab in
60". Training longer was tested at 90k and retired by a pre-registered
rate criterion (1 of 20 picked), although the distance distribution's low
tail keeps thickening with each doubling — a gain indistinguishable from
memorisation while no holdout existed.

## Stage 6 — Mechanism experiments: why does it miss?

All of these were **pre-registered** (decision log D48): the predictions
and decision rules were written down before the results existed.

- **Camera accounting — `harness_audit/camera_pixels.py`.** Aim: measure
  how many pixels the target occupies in each camera stream as a function
  of distance, reconstructed from logged expert states. Expectation: the
  external camera is weak. Result: the external camera carries 0–4 pixels
  of target at all distances (effectively a dead input), and the nose
  camera is thin exactly at the range where the policy commits to a
  direction.
- **Lateral tracking decomposition — `harness_audit/powered_corr.py` and
  `harness_audit/audit_analysis.py`.** Aim: regress the policy's
  behaviour against scene geometry to localise the failure. Result: the
  policy tracks the object's lateral position with a slope of only about
  0.4 (it flies a compromise between the target and its prior), and two
  early conclusions drawn from these analyses were later formally
  retracted when a reviewer showed one test was non-discriminating and
  one claimed effect was a kinematic identity — the retractions are part
  of the record.
- **Execution-horizon test (`--exech`).** Aim: if fifty-step open-loop
  chunks are the problem, re-planning more often should help.
  Expectation, honestly, was that it would help. Result: it made things
  dramatically worse (median 412 mm) — replanning interrupts the
  commitment the policy relies on, and "chunk length" was exonerated.
- **Oracle-yaw (`--oracleyaw`, `--oracleyaw2`).** Aim: physically pin the
  drone's heading onto the true object bearing while the policy controls
  everything else, to test whether heading ownership explains the miss.
  Result: the strongest intervention of the project — the median miss
  collapsed from 242 mm to **38.6 mm at n=40** (two seeds pooled, 6/40
  picked), a 6.3× reduction, with the stated caveat that pinning heading
  changes aiming, camera view, and gripper geometry at once.
- **Oracle-vector (`harness_audit/build_oracle.py`, `--oraclestate`,
  `--oraclecorrupt`).** Aim: append the exact gripper-to-target vector to
  the state, fine-tune briefly, and see whether *information* is what is
  missing; a paired control corrupts the vector's direction to prove the
  model actually reads the new input. Result: the control passed
  decisively (corrupting the vector costs +146 mm paired), so the model
  did attend to the feature — and even so it only reached median 168 mm
  with 1/20 picked. Perception was never the whole constraint: telling
  the policy exactly where the target is helps far less than physically
  acting on the heading. The bottleneck is on the action side.

## Stage 7 — Language: the probe and the correction

**File: `Sim'n'Real/Mujoco/forgetting_probe.py` (decision log D50).**

Aim: place a mustard bottle — an object π₀'s pretraining knows but this
project's tasks never used — against the trained weight, in a 2×2 design
(prompt names the bottle or the weight; positions normal or swapped), to
ask whether fine-tuning destroyed the model's ability to act on object
words. Expectation: if grounding survived, selection follows the prompt.
Result: selection followed **position** in all four arms (15, 15, 14 and
16 of 20 episodes to the task spot regardless of what was named); the
prompt noun is behaviourally inert. The mechanism I first recorded —
"training always put the prompted object in the same spot, so language
was never needed" — was an inference, and it was **falsified the next
day** by a direct measurement (below) and formally corrected in the log.

## Stage 8 — Dataset tests: five free experiments before any re-collection

**Files in `dataset_tests/`; reports `REPORT.md` and `REPORT2.md`. All
read existing data only — no GPU, no training. Every decision rule was
written before the numbers.**

- **`reversal_probe.py` — does the expert reverse before grasping?** Aim:
  a policy that executes fifty actions blind would struggle with a
  back-and-forth final approach, so measure whether the demonstrations
  contain one. Expectation: some reversals. Result: **none, 0/225** — and
  stronger, the commanded sideways motion in the final six seconds is
  exactly zero. The expert parks early and holds still.
- **`position_shortcut.py` — could position substitute for language?**
  Aim: measure whether any position rule identifies the target, which
  would make the prompt redundant in training. Expectation: yes (this was
  the recorded D50 mechanism). Result: **no — the best cross-validated
  rule scores 52.3%, chance.** The objects sit side by side with the
  target assigned to either side; language was the only cue, in every
  episode, and the model failed to learn it. This falsified my D50
  mechanism and moved the inert prompt from "collector flaw" to
  "model/training failure".
- **`hedging_probe.py` — is the policy aiming between the two objects?**
  Aim: if the policy cannot resolve which object is the target, it should
  end near the midpoint of the pair; the distractor positions are in no
  log, so the harness RNG stream was replayed and validated against the
  one immovable sampled quantity (60/60 bin positions reproduce to under
  a millimetre). Expectation: midpoint-dominant endings. Result: **the
  rule did not fire** — the gripper ends closest to the true target in
  32/60 episodes and closest to the midpoint in only 14, with a real but
  mild lean (median 22% of the way toward the distractor). Selection
  trouble is a contributor, not the dominant terminal mechanism. A
  secondary finding: the drone physically disturbs the task object in
  46/60 episodes.
- **`discrimination_render.py` — can the cameras tell the objects apart
  at all?** Aim: render both objects through all three cameras at the
  policy's 224-pixel input, at the distances where lateral commitment
  happens. Result: at 1.0–1.5 m **no stream separates them** — both are
  near-black smudges, colour never distinguishes them anywhere, and
  silhouettes only resolve inside about 0.6 m. This justified replacing
  the external camera *for discrimination*, explicitly recorded as a
  different claim from the earlier localisation one.
- **`parking_window.py` — how long is the freeze, exactly?** Aim:
  parameterise the v2 expert. Result: the last non-zero horizontal
  command comes a median **107 ticks (10.7 s)** before the grip closes —
  minimum 104, maximum 117, across all 225 episodes — longer than two
  full action chunks. Every demonstration taught "when close, stop and do
  nothing", which is precisely the behaviour the policy exhibits.

## Stage 9 — The v2 design (awaiting sign-off)

**Files in `v2_design/`: `camera_pose.py`, `terminal_profile.py`,
`CONFIRMATION.md` with rendered images.**

Every proposed change traces to one of the measurements above: a new
external camera pose measured to resolve both objects from every legal
spawn (24–104 pixels each versus 2–5 today, all five envelope scenes
passing, with the measured frame edge inducing a spawn envelope and
coin-flip target assignment so position stays uninformative by
construction); a taped-square goal on the table replacing the bin; a
terminal approach law that never stops moving (floor 3 mm/tick, grip
closing in motion, acceptance test: the parking window on collected data
must measure zero); 80 corrective episodes starting deliberately
off-target because arriving wrong and fixing it is a behaviour the
policy has never once seen; a widened start distribution; a 40-episode
holdout excluded from day one; and a plan to render the one collection
into two datasets differing only in the external camera, so the camera's
effect is attributable. Nothing is collected until the package is
confirmed.

## Verification and preservation (continuous)

**Files: `harness_audit/verify_ledger.py`, `eval_logs/VERIFICATION.md`,
`Reports/results_ledger.html`, git tags `results-v1-2026-08-30` and
`results-v2-2026-08-31`.**

Before the results page was rebuilt, every number on it was recomputed
from its raw log: 60 claims were checked, 52 verified as printed, and 8
corrected or re-labelled, with the corrections applied and the trace
committed. Checkpoints live on HuggingFace, raw logs and trajectory files
are committed under `Reports/eval_logs/`, the trajectory log is rotated
(never overwritten) at each protocol change, and the decision log records
retractions and corrections with the same prominence as findings.

---

## The through-line, in three sentences

The evaluation rig was broken and was repaired against a pre-committed
acceptance test before any conclusion about the policy was allowed to
stand. With an honest instrument, the failures localised not to
perception or to training scale but to the action side: a heading channel
the policy never exercises, a terminal freeze the data explicitly taught,
and a language channel the data required but the training never engaged.
The v2 collection changes exactly those things, each change traceable to
a number in this document, and trains with a holdout so the next round of
claims can be made without the memorisation asterisk.
