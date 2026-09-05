# AirVLA v2 — Complete Study Record

**Status: COMPLETE (frozen eval done 2026-09-05). This document is the
self-contained record of the v2 study: design, code, protocol, every
result, and the artifact index.** The chronological decision log lives
in `finaldroneresults.md`; corrections and redo-lessons in
`LESSONS.md`; this file is the organized reference.

---

## 1. Headline

π₀ (4.03B, flow matching) fine-tuned on 600 v2 simulator
demonstrations, evaluated closed-loop in the frozen v2 world
(replay-validated harness):

| | v1 (frozen n=60) | **v2 checkpoint 047500 (frozen n=60)** |
|---|---|---|
| Median closest jaw-to-target | 242 mm | **170.8 mm (−29%)** |
| Picked (weld + 12 cm lift) | 3 (latched-lift caveat) | 1 |
| **Placed (complete task)** | **0** | **1** (9.8 mm, weld tick 699, released in box, d_bin 29.8 mm) |
| Nav success | — | 9/20 (gate crossing **20/20**, 0 gate contacts) |
| Within 30 mm of target | — | 7/60 (11.7%) |

Total v2 policy grasps across all valid runs: **2** (both by 047500,
both scene 9 of seed family 97000): one grasp-and-carry (mini), one
complete pick-and-place (frozen eval). Videos:
`v2_eval_videos/grasps/`.

**The quantified bottleneck:** 7/60 episodes reach <30 mm but only 1
converts — terminal conversion under 50-step open-loop chunks, not
flight, approach, or harness. This motivates the E-track (§12).

---

## 2. System

- **Platform (sim):** SkyGrip quadrotor + 2-DoF arm + parallel-jaw
  gripper in MuJoCo; total sim mass 1.097 kg, hover thrust 10.76 N
  (PD flight controller, `pd_flight.py`; MPPI machinery present but
  collection/eval fly `flight="pd"`). Real-hardware reference masses:
  drone 736 g / arm 331 g / gripper 30 g (both sim models distribute
  mass differently — documented gap).
- **Physics pinning:** collection, training data, and all v2 evals on
  Myriad's mujoco 3.3.4 (glibc 2.17 ceiling). Sparks runs 3.11 —
  cross-version divergence measured and documented; never mix.
- **Model:** π₀ via LeRobot 0.6 (`lerobot-train` CLI, no custom
  trainer), base `lerobot/pi0_base` snapshot `25c379b52ba2…`,
  bfloat16, gradient checkpointing, vision encoder NOT frozen.
- **Action space (7-dim, per 10 Hz tick):** `[dx, dy, dz, 0, 0,
  dyaw, grip]` — world-frame setpoint deltas (|axis| ≤ SP_STEP
  0.035), droll/dpitch constant 0, yaw delta (|dyaw| ≤ YAW_STEP
  0.04), grip ABSOLUTE fraction 0–1 (slewed by GRIP_STEP 0.15
  commanded; fingers physically close ~0.8 mm/tick). MEAN_STD
  normalization; zero-variance dims safe (std+1e-8 → exact-0
  targets). Dataset action std: [0.0085, 0.0088, 0.0046, 0, 0,
  0.0146, 0.227].
- **Observations:** three 512×512 RGB cameras — camera1/2 (wrist
  stereo pair) + camera3 (fixed workspace view at (0, 2.55, 1.30),
  fovy 52, measured pose) — renamed at train time to pi0's
  base/left-wrist/right-wrist slots — plus proprioceptive state.

## 3. The v2 scene and choreography (why v2 exists)

v1's diagnosed defects: 107-tick parking pathology, position
shortcut ~52% (chance) confound, camera3 off-workspace, black penguin
(low contrast), box far from table. v2 redesign (every change traced
to a measurement or a directive from Hana):

- Box on the floor BESIDE the table (BIN_GAP 0.45 → ~22 cm clear
  gap); table FIXED at (0, 0.50) for constant camera-3 framing.
- Blue penguin (0.13, 0.33, 0.82) vs the wood table — dataset test B.
- Legs shortened to 35%; camera3 pulled back to keep table + box
  always in frame.
- Choreography: one continuous trajectory that speeds/slows (no
  pauses); ascend at spawn → yaw-in-place → straight transit →
  settle at 0.50 m standoff (arm still in travel pose) → re-aim →
  deploy arm to carry pose → 3 mm/tick creep with fire at
  CLOSE_FIRE_D (measured-finger-rate compensated) → weld on
  position+aperture gate → post-grasp stabilization → level carry
  with payload feed-forward + hold_xy trim → stable release (half
  GRIP_STEP + station hold) → recovery climb.
- Safety gates during collection (episode DISCARDED on violation):
  drone-table contact, phase-aware object strikes, gate strikes,
  pass-through/backup regrasp (frozen approach axis, forward-only
  ratchet, missed-close abort), arm-clearance ≥0.35 m from table
  when extending.
- Objects always on the table, 2-D positions randomized (x ±0.35,
  y 6–26 cm inside front edge for picks; full depth for nav),
  ≥0.40 m separation, target by coin flip (position uninformative BY
  CONSTRUCTION).
- Nav flavour: gate crossing at gripper-safe height then hover over
  the commanded object; balanced targets.

## 4. Code index (all v2 files)

| File | Role |
|---|---|
| `Sim'n'Real/Mujoco/collect_v2.py` | THE v2 collector: `V2Runner` (scene edits at runtime — no XML the frozen v1 eval imports is touched), expert choreography (`pick_v2`, nav), contact gates, payload feed-forward (`weld_grasp`), demo/demonav/collect modes, LeRobot dataset writer with atomic ffmpeg-concat patch + per-attempt manifest |
| `Sim'n'Real/Mujoco/collect_airvla.py` | v1 base runner/expert (imported, never edited — frozen-eval dependency) |
| `Sim'n'Real/Mujoco/collect_demos.py`, `pd_flight.py` | controller stack (SkyGripController, PD flight) |
| `Sim'n'Real/Mujoco/eval_v2.py` | Closed-loop eval in the v2 world: `V2Platform` (arm/grasp automaton), metrics, PROV with dep hashes, `--video` episode filming, per-episode payload-trim cleanup, contact tallies |
| `Sim'n'Real/Mujoco/replay_v2_platform_validation.py` | Ground-truth replay: expert actions through V2Platform must reproduce the expert's grasp (run before trusting ANY harness change) |
| `Sim'n'Real/Mujoco/valcurve_v2.py` | Validation curve: teacher-forced 50-step action MSE, 120 val episodes × 6 windows, pinned noise (31415) reused across checkpoints; resumable/incremental; T4 selection |
| `Sim'n'Real/Mujoco/v2_split.py` | Episode-level stratified 480/120 split (seed 424242, disjointness asserted) |
| `Sim'n'Real/Mujoco/v2_dataset_stats.py` | Section-C stats + pre-registered gate verdicts |
| `cluster_jobs/*.job`, `cluster_jobs/v2train_wrapper.py` | Banked SGE jobs: collection, trial, training (+wrapper for the node-local '/c' PermissionError), resume, 60k extension, validation sweep, rolling-validation daemon |
| `dataset_tests/` | v1-era dataset probes (reversal, position shortcut) with REPORTs |
| `v2_design/` | CONFIRMATION.md (design sign-off + amendments + demo verdicts), camera_pose.py, terminal_profile.py |

## 5. Dataset

- **600 episodes, zero rejections in 600 attempts** (17.8 h, job
  258624): 270 standard pick + 90 corrective (displaced-start) + 240
  nav. Objects balanced (std 132 weight / 138 penguin; nav 120/120).
  Episode lengths 144–665 ticks (mean 399).
- **Pre-registered gates PASS:** parking window median 1 / max 1
  tick (v1: 107); position shortcut 50.2% CV = exact chance.
- **Split:** 480 train / 120 val (54 std + 18 corr + 48 nav),
  episode-level, seed 424242 — `eval_logs/v2_split.json`.
- **Published:** `hanapasta/airvla_v2`, revision `2e40a5c89746`,
  read-back verified. Manifest:
  `eval_logs/v2_manifest_71000.jsonl` (every attempt: kind, object,
  coordinates, ticks, gate counters, outcome).
- Infrastructure lessons hit during collection (all documented in
  LESSONS/finaldroneresults): av-library incompatibilities → ffmpeg
  CLI concat monkeypatch; output-over-input truncation → atomic
  temp+replace+size guard; HF_LEROBOT_HOME → Scratch (quota).

## 6. Training

- **Protocol (frozen before results):** batch 4, seed 1000,
  save_freq 2500, MEAN_STD, camera rename map, no wandb, no hub
  push; train list = the 480-episode split via `--dataset.episodes`.
- **Run 1:** 30k steps (job 268762 + resume 280833 lineage; quota
  incident at step 5000 → janitor keeps newest 2 training_states of
  ~14 GB each). Loss 0.28 → ~0.08. 1.22 s/step on L-series.
- **Extension (amendment, Hana, 2026-09-04):** 30k → 60k (job
  280833) because validation was still descending; resumed from
  030000 with `--steps=60000` override (verified in the config dump:
  steps 60000, resume True, data order continuous at sample
  120,000); LR schedule already at decayed floor (decay_steps stayed
  30000) — the −23% further validation gain happened AT the floor.
- Training logs: `v2train_resume.log`, `v2train60k.log` (tqdm
  per-step loss = raw data for the loss-curve plot).

## 7. Validation curve and selection (T3/T4/T5)

Teacher-forced 50-step action MSE, real units, 120 held-out
episodes × 6 windows = 720 windows, flow noise pinned (seed 31415)
and reused for every checkpoint (exactly paired). Full 24-point
curve (`eval_logs/v2_valcurve_60k.json`):

| ckpt | MSE | ckpt | MSE |
|---|---|---|---|
| 2.5k | 0.000590 | 32.5k | 0.000062 |
| 5k | 0.000346 | 35k | 0.000055 |
| 7.5k | 0.000244 | 37.5k | 0.000056 |
| 10k | 0.000217 | 40k | 0.000053 |
| 12.5k | 0.000121 | 42.5k | 0.000054 |
| 15k | 0.000119 | 45k | 0.000052 |
| 17.5k | 0.000128 | **47.5k** | **0.0000455 ← selected** |
| 20k | 0.000098 | 50k | 0.000050 |
| 22.5k | 0.000073 | 52.5k | 0.000049 |
| 25k | 0.000061 | 55k | 0.000048 |
| 27.5k | 0.0000594 | 57.5k | 0.000051 |
| 30k | 0.000064 | 60k | 0.000055 |

- **T4 selection: 047500** (lowest overall val MSE; rule frozen
  before any results; genuine bracketed minimum — both neighbours
  worse, gentle upturn to 60k).
- Reading: fast fit to 12.5k, then a slow halving to 47.5k even at
  floor LR; single points wobble ~10% (17.5k blip) — only the full
  curve is trustworthy. Ranking carried closed-loop signal (only
  the minimum ever grasped) though magnitudes don't predict
  closed-loop success (compounding deviation).
- per_dim (7) + per_flavour (std/corr/nav) recorded for every
  checkpoint from 15k on; grip dim shrank 5.3e-4 → 2.1e-4
  (15k→27.5k), all flavours improved together.
- Selected checkpoint published:
  `hanapasta/airvla_v2_pi0_047500` (private), revision
  `a405a487f891`, read-back verified.

## 8. Evaluation harness (and how it was validated)

`eval_v2.py`: policy flies closed-loop from its three cameras +
state; 50-step chunks executed open-loop then re-planned (naive
mode = the frozen baseline); scene seed 97000 (disjoint family),
torch seed 1000; PROV line hashes the script AND its physics
dependencies (collect_airvla, collect_v2, pd_flight, collect_demos).

`V2Platform` (the scripted body layer — arm + weld, policy owns
flight/grip):
- **Arm deploy:** proximity-debounced (dxy < 0.60 m to the live
  target for 5 consecutive ticks) — mirrors the expert (deploys
  after settling at the 0.50 m standoff), NEVER on ascent.
- **Weld gate (collector-verbatim):** jaw-to-aim <10 mm horizontal
  AND <15 mm vertical AND aperture inside the PER-OBJECT window
  (`cur["ap_lo"] < ap·1000 < cur["ap_hi"]`); no contact term.
- **Release:** policy opens grip (>0.8) while welded. Payload
  feed-forward on weld/release; episode-end `weld_grasp(False)`
  cleanup if still latched (cross-episode trim-leak fix).
- **Contact observations:** per-tick drone-table, strike-object,
  drone-gate tallies (collector cadence/dedupe; jaws-in-grasp-phase
  rule deliberately omitted — the policy owns grasp timing).

**Ground-truth replay validation (2026-09-04, the decisive QA
step):** expert actions through the platform on a same-seed twin
runner. Found TWO v1-heritage defects (deploy at tick ~5 vs expert's
~110; unsatisfiable pad-contact weld gate — v2's close-in-motion
grasp welds before pads register). Post-fix: **3/3 episodes (weight
+ 2 penguins) weld at the expert's exact tick (292/195/241), lift
+223/+475/+262 mm, release cleanly, no cross-episode leak.** All
pick metrics measured BEFORE this fix are void as policy
measurements (mini30k, mini30k_r2). Script banked:
`replay_v2_platform_validation.py`. Lesson: replay expert actions
through ANY new/changed harness before reading policy results — it
caught what adversarial code review passed.

## 9. All evaluation results (valid runs)

Seed family 97000 throughout; episodes i share scenes across runs
(paired). "picked" = welded + object >12 cm; "placed" = in box.

| Run (job, tag) | Ckpt | Picks | Nav | Notes |
|---|---|---|---|---|
| mini30k_r3 (282168) | 030000 | n=10, median 98.8 mm, 0 picked | 3/4 (crossed 4/4) | first VALID pick measurement; all contact-clean |
| mini47k5 (283113) | 047500 | n=10, median 175.5 mm, **1 picked** (ep9: 9.3 mm, weld 732, carried) | 2/4 (crossed 4/4) | first v2 grasp ever |
| **full47k5 (283261, FROZEN)** | **047500** | **n=60, median 170.8 mm, 1 picked, 1 PLACED** (ep9: 9.8 mm, weld 699, d_bin 29.8 mm) | **9/20 (crossed 20/20, 0 gate contacts)** | 7/60 <30 mm; 55/60 contact-clean (worst: ep10 31 table-ticks/22 obj; ep35 22/3) |

Void (broken-harness era, kept for the record): mini30k (277519) and
mini30k_r2 (280417) — platform could not weld and flew arm-out from
tick ~5; nav rows remain valid (nav never deploys/welds).
Determinism probes: mini30k_r2 reproduced mini30k within 1–3 mm
early / tens of mm late (cross-node jitter amplification, judged by
onset per the pre-registered rule).

Full miss distribution (full47k5, mm, sorted): 9.8✓ 17.8 20.1 24.9
25.3 25.9 27.6 | 35.7 40.7 46.8 55.7 60.8 64.6 65.9 73.0 76.8 82.1
87.3 90.1 95.3 | 105.2 109.2 121.5 123.9 127.9 128.6 146.6 159.8
161.6 169.6 | 170.7 170.8 175.5* … up to 613.9 (3 episodes >550).
(*script median convention mm[n//2] = 170.8.)

## 10. Seed registry

collection 71000 · trial 72000 · split 424242 · training 1000 ·
v1 eval 77000 · probes 88000 · demos 31–33000 · v2 eval scenes
97000 · v2 eval torch 1000 · valcurve noise 31415.

## 11. Artifact index

- **Ledger:** `Reports/finaldroneresults.md` (pre-registration +
  dated results + incidents). Checklists: `EVIDENCE_CHECKLIST.md`,
  `DISSERTATION_CHECKLIST.md`. Lessons: `LESSONS.md`;
  `TEST_FILES.md`, `EXPERIMENT_CHRONICLE.md`.
- **Videos:** `Reports/v2_eval_videos/` — mini30k_r2/r3 + mini47k5
  sets (14 each), `full47k5/` (all 80, size-verified),
  `grasps/` (the two grasp episodes), contact sheets (pick00,
  nav01, r3 pick03, mini47k5 pick09, full47k5 pick09 success).
  Demos: `v2_demo.mp4`, `v2_nav_demo.mp4`, `v2_trial_cam3.mp4`.
- **Data/JSON:** `eval_logs/v2_manifest_71000.jsonl`,
  `v2_split.json`, `v2_valcurve_60k.json`,
  `eval_v2_traj_mini_20260904.jsonl` (30k_r2),
  `eval_v2_traj_mini_20260905.jsonl` (30k_r3 stack),
  `eval_v2_traj_mini47k5.jsonl`, `eval_v2_traj_full47k5.jsonl`.
- **HF:** dataset `hanapasta/airvla_v2` (2e40a5c89746); checkpoint
  `hanapasta/airvla_v2_pi0_047500` (a405a487f891); trial
  `airvla_v2_trial` (d74f6c5e9a49).
- **Myriad:** `~/Scratch/airvla/sim/` (scripts),
  `pi0_v2_out/checkpoints/` (002500–060000), logs
  `~/Scratch/airvla/logs/` (v2collect_258624, v2train_resume,
  v2train60k, v2valsweep, v2val60k, v2mini30k_r2/r3, v2mini47k5,
  v2full47k5, hfup47k5).
- **Key commits:** collector approved a313575→d6b5339; eval harness
  b164627 (video+counters), dbea3bc (FF-leak fix), f7ec4ee (replay
  fixes); results 2034da6, 1347493, c32cc24, 629187f, 294c418,
  47c60d8, 78cc8a2.

## 12. Recreation recipe (exact commands)

```bash
# 1. Collect (Myriad, mujoco 3.3.4, MUJOCO_GL=egl, HF_LEROBOT_HOME on Scratch)
python collect_v2.py collect hanapasta/airvla_v2 71000 20   # units → 600 eps

# 2. Split + stats + gates
python v2_split.py            # → v2_split.json (480/120, seed 424242)
python v2_dataset_stats.py    # gates: parking ≤1 tick, shortcut ~50%

# 3. Train (see cluster_jobs/v2train_c.job + v2train_wrapper.py)
lerobot-train --dataset.repo_id=hanapasta/airvla_v2 \
  --dataset.episodes=[train list] --policy.path=<pi0_base snapshot> \
  --rename_map='{"observation.images.camera3":"observation.images.base_0_rgb",
    "observation.images.camera1":"observation.images.left_wrist_0_rgb",
    "observation.images.camera2":"observation.images.right_wrist_0_rgb"}' \
  --output_dir=…/pi0_v2_out --batch_size=4 --steps=60000 \
  --save_freq=2500 --num_workers=0 --seed=1000 \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false
# extend/resume: --config_path=<ckpt>/last/pretrained_model/train_config.json --resume=true --steps=60000

# 4. Validation curve + T4 selection (resumable)
python valcurve_v2.py <ckpt_root> v2_split.json v2_manifest_71000.jsonl v2_valcurve.json

# 5. Validate the harness BEFORE any eval (must weld 3/3 at expert ticks)
python replay_v2_platform_validation.py

# 6. Frozen eval of the selected checkpoint
python eval_v2.py <ckpt>/047500/pretrained_model 60 20 \
    --torchseed 1000 --tag full47k5 --video
```

## 13. Known limitations

- 50-step open-loop chunk execution (naive mode) is the binding
  constraint — the E-track below addresses it.
- Corrective flavour covers displaced STARTS, not terminal
  misalignment (the state the policy actually fails in).
- Validation MSE ranks checkpoints but does not predict closed-loop
  success magnitudes (compounding deviation).
- Cross-node GPU jitter makes per-episode results reproducible only
  to a few mm early / tens of mm late in an episode; judge re-runs
  by divergence onset.
- Sim-only; real-hardware mass distribution differs (see §2).

## 14. Next phase — E-track (pre-registered in finaldroneresults.md)

Target: **≥60% grasp (picked) rate on the frozen n=60**. Evidence
base: LIBERO π0.5 75%→90% from execution-horizon 50→10; RTC gains in
the AirVLA paper itself (gate 50→80% / 45→95%, payload-aware
guidance up to +50% aerial grasping); terminal-corrective data
(DAgger family); residual-RL fine-tuning (+34 pts typical).
Conditions: E1 exec-horizon 10 · E2 RTC · E3 terminal-corrective
fine-tune · E4 test-time sampling · E5 residual RL (escalate only as
needed). Naive-50 full47k5 remains the frozen baseline forever.
