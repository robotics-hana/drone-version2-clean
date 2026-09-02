# Final drone results — v2 campaign tracker

Started 2026-09-01. This file is the single tracker for the v2 dataset →
training → evaluation campaign: pre-registered definitions first (locked
BEFORE collection/training), then results as bullets as they land.
Nothing in the pre-registration section may be edited after training
starts — corrections get dated addenda.

## The questions (defined before collecting)

- Q1: Is the dataset sufficiently diverse and physically valid?
- Q2: Does π0 converge and generalise to held-out validation episodes?
- Q3: Does additional demonstration data improve adaptation?
- Q4: Does RTC improve over naive execution?
- Q5: Does PAG improve over RTC?
- Q6: Does PAG specifically reduce payload-induced disturbance?
- Q7: Does the effect increase under the heavier payload?
- Q8: Does the policy generalise to unseen objects/configurations?
- Q9: Is the improvement statistically meaningful?
- Q10: Is the resulting system computationally viable?

## A. Pre-registered randomisation (the training distribution)

Collector: `collect_v2.py` (mode `collect`), seed 71000, single Myriad
job, mujoco 3.3.4 (train physics = frozen-eval physics; v1 had a hidden
3.11-data/3.3.4-eval mismatch).

| Variable | Distribution |
|---|---|
| Table | FIXED at (0.00, 0.50), half-extents (0.45, 0.30) — constant camera-3 framing by design |
| Object positions (both) | 2D on-table: x ~ U(−0.35, +0.35); y ~ U(0.54, 0.74) for picks (grasp-reachable band), U(0.28, 0.74) for nav (full depth); pair separation ≥ 0.40 m |
| Target assignment | COIN FLIP between the two sampled spots (position uninformative by construction) |
| Object identity | Coin flip weight / blue penguin per episode |
| Object orientation | Random yaw per object (collector's per-episode quaternion draw); task-penguin beak constrained into the table (v1 rule, grasp safety) |
| Box | Beside the table, side coin-flipped (±(0.45+0.45), y = 0.50), squared, on the floor |
| Box/goal tints | v1 tint randomisation unchanged (wood tone + pale distractor box) |
| Gate (nav) | Side coin-flipped x = ±0.7, y = −0.6; crossing height 0.95 |
| Drone start (pick) | x ~ U(−0.5, 0.5), y ~ U(1.1, 1.7), z ~ U(0.21, 0.60), ≥ 0.70 m from target |
| Drone start (nav) | x ~ U(−0.95, 0.95), y ~ U(−1.9, −1.2), z ~ U(0.60, 1.00), yaw π ± 0.26 |
| Task variation | 3 flavours: standard pick, corrective pick (displaced approach, U(5, 20) cm at room-side bearing), nav (gate + hover) |
| Payload | Object masses as modelled (weight vs penguin); 45 g / 100 g payload study is a separate later experiment (PAG track) |

## Composition and split (Hana, 2026-09-01)

| Task | Episodes | Training | Validation |
|---|---:|---:|---:|
| Pick-and-place (270 standard + 90 corrective) | 360 | 288 | 72 |
| Navigate + hover | 240 | 192 | 48 |
| **Total** | **600** | **480** | **120** |

- Split by COMPLETE episodes, stratified by flavour (corrective split
  72/18 inside the pick quota), assigned by seeded RNG (seed 424242) on
  banked episode indices; committed as `Reports/eval_logs/v2_split.json`
  before training.
- Frames from one episode never straddle splits. No episode duplication
  (unique seeds by construction — one RNG stream, no reuse).
- The **test set is not drawn from these 600**: testing is the frozen
  simulator evaluation (fresh seeded scenes, protocol v1/v2 scripts,
  seed families disjoint from collection), plus the independent probes
  below. Test scenes never influence training, checkpoint selection, or
  any parameter.

## Pre-registered OOD conditions (decided BEFORE training)

1. **OOD object**: the mustard bottle — excluded from every v2 episode;
   probe = the existing 2×2 forgetting-probe design on the v2 policy.
2. **OOD spatial (start)**: pick-task drone spawns z ∈ (0.70, 1.00) —
   entirely outside the training U(0.21, 0.60).
3. **OOD spatial (object)**: target x ∈ ±(0.38, 0.44) — on the table but
   outside the trained ±0.35 band.
4. **OOD instruction**: the held-out prompt phrasing (existing
   PROMPT_HELDOUT mechanism).
5. **Held-out compositional task**: gate → pick → place (comp) — never
   collected in any flavour; evaluated only at test time.

## B. Episode records + acceptance gates (enforced by the collector)

Manifest (`v2_manifest_71000.jsonl`, one line per ATTEMPT, banked or
rejected): attempt #, slot, kind, collector seed, object, target/spot
coordinates, gate side, tick count, d_bin / crossed / hover, parking
window, table_hits, obj_hits, gate_hits, banked, rejection reason.

Acceptance requires ALL of: physically stable throughout (no tumble —
covered by task completion + contact gates) · genuine grasp (weld only
from measured seated pinch inside the aperture window — no visual-overlap
artefact possible) · object attached through transport (weld released
only at the box) · placement genuinely inside the box (d_bin ≤ 150 mm =
inside walls) · zero drone–table contacts · zero non-grasp drone–object
contacts · zero gate contacts (nav) · correct termination. Rejected
episodes are retained in the manifest as evidence.

## E. Leakage prevention

- Collection seed family (71000) disjoint from eval scene families
  (77000-series), probe families (88000), demo families (31–33000).
- Mustard bottle, OOD spatial bands, comp task: excluded by construction
  (see pre-registration above), not by post-hoc filtering.
- Normalisation statistics computed by LeRobot from the TRAIN split only
  (`--dataset.episodes` passes the train list; MEAN_STD fitted on what
  the trainer sees).
- `position_shortcut.py` re-run on the collected data before training
  (pass = no shortcut > 55%); `parking_window.py` re-run (pass = ≤ 1
  tick median).

## T. Training protocol (frozen before results)

- Trainer: stock `lerobot-train`, π₀ from `lerobot/pi0_base`, full
  fine-tune (vision encoder unfrozen, expert not isolated).
- Batch 4, 30,000 steps, AdamW lr 2.5e-5 (1k warmup, cosine → 2.5e-6),
  weight decay 0.01, bfloat16, gradient checkpointing, save every 2,500.
- Seed 1000. Hardware: Myriad A100. Dataset version: the HF revision
  hash of `hanapasta/airvla_v2` recorded at submission.
- **Checkpoint selection criterion (T4, declared now)**: lowest
  teacher-forced action MSE on the 120 validation episodes, evaluated
  for every saved checkpoint; never revised against test results.
- T5 per-dimension validation action error reported for all 7 dims.
- Scaling (T7) and corrective-ablation (T8) runs use the same frozen
  protocol with `--dataset.episodes` sublists.

---

# RESULTS (bullets, appended as they land — never edited, only added)

## Collection

- **2026-09-02 00:2x — Myriad physics validation PASSED** (job 257109,
  mujoco 3.3.4): picks 3/3 grasped+placed (18.2/35.0/36.9 mm), nav 6/6
  crossed+hovered, zero contacts on every gate in all 9 episodes.
  Distances differ from the Sparks (3.11) validation as expected —
  cross-version physics drift — but every acceptance gate holds on the
  collection platform. Collection authorized on Myriad; train physics
  will equal frozen-eval physics.
- **2026-09-02 — collection job submitted**: single Myriad GPU job,
  `collect_v2.py collect hanapasta/airvla_v2 71000 30` (600-episode
  balanced plan), manifest `v2_manifest_71000.jsonl`.
- **2026-09-02 — three false starts before the clean run, all
  infrastructure, zero lost data**: (1) av-library incompatibility in
  the Myriad writer (glibc ceiling blocks av≥15; av 14/13 each break a
  different lerobot API — fixed with an ffmpeg-CLI concat patch,
  smoke-tested before use); (2) LeRobot's default dataset home is the
  40 GB HOME quota — repointed to Scratch; (3) a stale-log/duplicate-
  job tangle caused by the job template reusing one log file — the
  never-reuse-a-log rule now applies to job logs too (per-job
  filenames), and the watcher verifies the patch marker + exactly one
  running job at startup.
- **2026-09-02 04:0x — MILESTONE: 100/600 banked in 100 attempts — 100%
  banking yield, zero rejections through the first hundred episodes.**
  Job 257145, ETA ~14 h to completion.
- **2026-09-02 13:1x — RUN 257145 KILLED AT 440/600: video corruption
  found by a mid-run content check.** The ffmpeg concat patch wrote its
  output over one of its own inputs (append passes output==input[0]);
  stream-copy onto a file being read truncates silently — 440 episodes
  of parquet were perfect while the video chunks held ~100 KB. Caught
  before training, not after. Fix: concat to a temp sibling, verify
  size ≥ half the inputs, atomic os.replace; re-validated by a
  content-probing smoke (75/75 frames, clean decode). Lesson appended
  to LESSONS territory: **smoke tests must verify content, not exit
  codes** — the original smoke printed "saved OK" over truncated
  output. Attempts and rejection stats to this point: 440/440 banked,
  zero gate rejections (the choreography itself is flawless; every
  failure this campaign has been infrastructure).

## Dataset checks (C/E)

- **2026-09-02 — PIPELINE TRIAL: all five stages verified end-to-end on
  a 20-episode trial set (seed 72000, `airvla_v2_trial`) before the real
  data depends on them.** (1) Collection 20/20 banked, zero rejections.
  (2) Stats + gates: parking window median 1 tick (max 1) PASS;
  position shortcut 37.4% CV PASS; spatial coverage spans the design
  bands. (3) Episode-level stratified split 15/5, disjointness asserted.
  (4) HF push verified by READ-BACK (9 files, revision d74f6c5e9a49) —
  not by exit code. (5) 50-step training smoke consumed the split's
  actual train list + camera rename map and saved checkpoint 000050,
  exit 0. (First smoke attempt was wall-clock killed at 1 h during a
  contended model load — rerun with 3 h passed; budget lesson applied
  to the real training job.)
- **2026-09-02 17:0x — main collection 100/600, 100-for-100, video
  content spot-check 2.0 GB at 100 episodes** (the corrupt run held
  100 KB at 440 — the atomic-concat fix is proven on real data).

## Training

- (pending)

## Evaluation

- (pending)
