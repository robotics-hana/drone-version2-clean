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

## Collection — FINAL

- **2026-09-03 07:0x — COLLECTION COMPLETE: 600/600 banked in 600
  attempts. ZERO rejections across the entire campaign.** 17.8 h, job
  258624, clean exit. Composition exactly 270 std / 90 corr / 240 nav;
  objects balanced (std 132 w / 138 p; nav 120/120; corr 57/33 —
  coin-flip noise). Spatial coverage spans the full design bands with
  centred means. Episode lengths 144–665 ticks (mean 399).
- **Both pre-registered gates PASS on the real dataset**: parking
  window median 1 tick with MAX 1 across all 360 pick episodes (the v1
  pathology this campaign exists to fix was 107); position shortcut
  50.2% CV = exact chance (v1's confound, now impossible by
  construction). Manifest + split banked in `Reports/eval_logs/`.
- Split: 480 train / 120 validation — exactly the specified 54 std +
  18 corr + 48 nav quotas, episode-level, disjointness asserted.
- Dataset pushed to `hanapasta/airvla_v2` (read-back-verified; revision
  recorded with the push log).

## Training

- **2026-09-03 — 30k training SUBMITTED (job 268762)** under the frozen
  protocol: batch 4, seed 1000, 24 h wall (smoke lesson), train list =
  the 480-episode split, rename map camera3→base / camera1→left-wrist /
  camera2→right-wrist, output `pi0_v2_out`.

## Evaluation

- **2026-09-04 — validation sweep (T3/T4) in progress, job 276984**
  (resumable rerun after a 4 h wall kill at 5/12): teacher-forced
  50-step action MSE, 120 val episodes × 6 windows = 720 windows,
  noise pinned (seed 31415) and reused across checkpoints. Curve so
  far: 2.5k 0.000590 → 5k 0.000346 → 7.5k 0.000244 → 10k 0.000217 →
  12.5k 0.000121 → 15k 0.000119 → **17.5k 0.000128 (TURNED UP —
  overfitting inflection past 15k; training loss still falling)**.
  Provisional T4 winner 015000, pending the full curve.
- **2026-09-04 — mini closed-loop test of the LAST checkpoint (030000),
  job 277519, eval_v2.py, seed family 97000, tag mini30k**: NAV 3/4
  success (clean gate crossings + hover); PICK n=10 median miss
  215.6 mm, 0 picked (distribution 47–589 mm) — statistically
  indistinguishable from v1's 242 mm at n=10. Tiny validation MSE +
  closed-loop misses = compounding-deviation signature; the full n=60
  frozen eval of the SELECTED checkpoint is the referendum. (First
  mini attempt was INVALID — no arm platform, jaws tucked — killed,
  V2Platform ported from the fixfit5-validated v1 path, rerun.)
- **2026-09-04 — eval harness defect found by pre-run adversarial
  review and FIXED**: eval_v2's table/obj/gate contact counters were
  structurally 0 (the collector tallies them inside V2Runner.step,
  which eval never calls) — the mini30k rows' contact fields are VOID
  (primary metrics unaffected); the tally is now replicated per-tick
  in V2Platform.tick (jaws-in-grasp-phase rule deliberately omitted —
  the policy owns its grasp timing). Also added `--video` (per-episode
  cam1|cam2|cam3 strips, the demo-video format).
- **2026-09-04 — job 279988 submitted**: mini test of checkpoint
  015000 (10 pick + 4 nav, tag mini15k, SAME scenes/seeds as mini30k →
  exactly paired checkpoint comparison) + 3 filmed episodes of 030000
  (tag film30k, scenes 0–2 of the mini30k sequence — should reproduce
  its per-episode misses exactly, a free determinism check). Both with
  --video; log v2mini15k_film.log.
- **2026-09-04 — SECOND review-caught harness defect, KILL-FIX-REDO of
  279988 → job 280071**: cross-episode payload feed-forward leak — an
  eval episode that ends still CARRYING leaves the payload trim
  latched on nominal_hover_thrust (weld_grasp adds it at weld, only a
  release removes it; the collector never exposes this because
  place_v2 always releases). Every later episode in the process then
  flies a mis-trimmed plant, keyed to that checkpoint's own behaviour
  → paired comparison broken from the first carried-to-cap episode.
  Fixed: episode-end `if r._ff_on: weld_grasp(False)` in run_pick;
  weld now OBSERVABLE (weld_tick + ended_welded in every pick row —
  previously unlogged, which is why mini30k cannot be certified
  leak-free post hoc); nav rows now report obj_hits; PROV extended
  with dep_sha of collect_airvla/collect_v2/pd_flight/collect_demos
  (drift audit). 279988 killed before its first episode (nothing
  lost). Job 280071 redoes BOTH minis under the fixed harness, 10+4
  each, tags mini15k_r2 / mini30k_r2, both filmed, log v2mini_r2.log.
  mini30k (277519) is now PROVISIONAL — superseded by mini30k_r2.
  Determinism probe: mini30k_r2 picks 0–9 re-see mini30k's exact
  scenes+noise; same-node ⇒ near-exact reproduction expected, cross-
  node ⇒ small divergence is hardware noise (judge by onset, per the
  determinism review), contact/weld fields not comparable (were dead).
- **2026-09-04 13:3x — 15k mini CANCELLED (Hana), 280071 killed in
  model load (no episodes lost)**: the sweep's 20k point (0.000098,
  new best) made 15k unlikely to be the T4 winner. Job 280417 runs the
  30k-only mini redo (10+4, tag mini30k_r2, filmed, fixed harness);
  the paired mini of the SELECTED checkpoint runs after T4 selection.
