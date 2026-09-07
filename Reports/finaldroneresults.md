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
- **2026-09-04 15:0x — mini30k_r2 COMPLETE (job 280417, exit 0), all
  14 episodes filmed**: PICK n=10 median 238.4 mm, 0 picked, 0 placed
  (47.9 / 68.7 / 110.0 / 207.4 / 225.5 / 238.4 / 273.9 / 284.1 /
  410.4 / 589.6); NAV 3/4 (crossed 4/4, one hover miss — same episode
  pattern as the original). **Original mini30k CERTIFIED leak-free**:
  every pick ended_welded=false / weld_tick=null, so the payload-FF
  leak could never have fired in it — its numbers stand (promoted from
  provisional). **Determinism probe**: early episodes reproduce the
  original within 1–3 mm, later ones drift by tens of mm (compounded
  cross-run jitter, onset-consistent with the pre-registered
  hardware-noise interpretation; medians 215.6 → 238.4). **Contact
  observations now real and CLEAN**: 0 table hits, 0 object strikes,
  0 gate hits across all 14 episodes — the 30k policy misses grasps
  but flies clean. Videos: `Reports/v2_eval_videos/` (14 mp4, cam
  strips; content spot-checked via contact sheets — pick00 shows the
  47.9 mm near-miss then continue-to-box; nav01 shows a clean gate
  crossing + target hover). Traj:
  `Reports/eval_logs/eval_v2_traj_mini_20260904.jsonl`.
- **2026-09-04 22:0x — GROUND-TRUTH REPLAY falsifies the v2 eval
  platform; TWO defects fixed and validated; ALL prior v2 closed-loop
  PICK metrics VOID as policy measurements.** The v1 lesson (replay
  expert actions through the eval harness before reading policy
  results) finally ran in the v2 world
  (`replay_v2_platform_validation.py`, local, seed 97000) and showed
  the expert's own recorded actions COULD NOT WELD through
  V2Platform: (1) the v1-heritage arm-deploy rule (sp_z≥0.45 or tick
  60) fired at tick ~5 vs the expert's ~110 (the expert deploys only
  after settling at the 0.50 m standoff) — the early swing destroyed
  the approach (5.8 m excursion, ~200 mm standing offset through the
  grasp window); (2) the v1-heritage weld gate (fixed 4–17 mm
  aperture + pad-contact 2-of-4) is UNSATISFIABLE in v2 — the
  close-in-motion grasp welds on proximity before pads register
  contact (expert gate: 10 mm horiz / 15 mm vert to aim + PER-OBJECT
  aperture window, no contact term). Fixes: proximity-debounced
  deploy (dxy<0.60 for 5 ticks) + collector-verbatim weld gate.
  Post-fix replay: 3/3 episodes (weight + 2 penguin) weld at the
  expert's exact tick (292/195/241), lift +223/+475/+262 mm, release
  cleanly, no cross-episode leak. CONSEQUENCE: mini30k and mini30k_r2
  pick rows (0 picked, miss medians 215.6/238.4 mm) were measured
  under a platform that could not weld and flew arm-out from tick ~5
  — VOID as policy grasp measurements (nav rows unaffected: nav never
  deploys/welds). Job 282168 re-runs the 30k mini (10+4, tag
  mini30k_r3, filmed) under the validated harness.
- **2026-09-05 00:5x — mini30k_r3 COMPLETE (job 282168, exit 0): the
  first VALID v2 closed-loop pick measurement.** PICK n=10 median
  98.8 mm, 0 picked, 0 placed — misses 43.5 / 55.3 / 65.9 / 73.1 /
  77.9 / 98.8 / 237.8 / 534.4 / 542.0 / 553.3; NAV 3/4 (same episode
  pattern). Read: under training-matched dynamics the approach
  improves ~2.4× (median 238→99 mm; 6/10 episodes within 100 mm,
  distribution bimodal with 3 far misses ~540 mm), but the policy
  still never enters the 10 mm weld zone — the video of the closest
  episode (pick03, 43.5 mm) shows the jaws beside the penguin,
  laterally offset a few cm, before the routine proceeds to the box.
  Failure is now cleanly the LAST-DECIMETER SERVO, not flight or
  harness: consistent with the v1 precision-gap finding and the
  compounding-deviation signature (50-tick open-loop chunks cannot
  correct terminal misalignment the way the expert's every-tick creep
  servo does; corrective demos cover displaced starts, not terminal
  offset). Videos `Reports/v2_eval_videos/v2vid_mini30k_r3_*.mp4`
  (14), sheet of pick03 banked; traj
  `Reports/eval_logs/eval_v2_traj_mini_20260905.jsonl`. Next
  referendum: full frozen n=60+20 eval of the 60k-curve T4 winner
  under this validated harness.
- **2026-09-05 04:5x — 60k TRAINING + FULL VALIDATION CURVE COMPLETE;
  T4 SELECTION: checkpoint 047500, val action-MSE 0.0000455.**
  Training (job 280833) exited 0 at step 60,000; rolling daemon (job
  280834) scored every checkpoint with the pinned-noise paired
  protocol and self-terminated. Full curve (24 points,
  `eval_logs/v2_valcurve_60k.json`): 0.000590 (2.5k) → 0.000119
  (15k) → 0.0000594 (27.5k) → 0.0000528 (40k) → **0.0000455
  (47.5k, minimum)** → 0.000048–0.000055 (50k–60k, mild upturn).
  The extension verdict: 30k→47.5k bought a further −23% validation
  error at floor LR; past 47.5k the curve turns gently up — a
  genuine, well-bracketed minimum. Winner has full per-dim +
  per-flavour data (T5). Job 283113 runs the winner's mini (10+4,
  tag mini47k5, filmed, validated harness) as the pre-eval smoke;
  the full frozen n=60 pick + 20 nav evaluation of 047500 follows a
  clean mini.
- **2026-09-05 06:1x — mini47k5 COMPLETE (job 283113, exit 0):
  FIRST CLOSED-LOOP GRASP OF THE v2 CAMPAIGN.** Episode 9: miss
  9.3 mm, weld at tick 732, object lifted and carried to episode end
  (picked=true; not placed — carried away from the bin). On film:
  `v2vid_mini47k5_pick09.mp4` (grasp-moment sheet banked). Full
  results: PICK n=10 median 175.5 mm, 1 picked, 0 placed (9.3 /
  63.6 / 66.1 / 110.8 / 161.7 / 175.5 / 303.5 / 435.4 / 482.2 /
  613.9); NAV 2/4 (crossed 4/4, two hover misses). Paired read vs
  30k (same scenes/noise/harness): 047500 converts one near-miss
  scene into a grasp; per-scene misses track 30k's pattern
  otherwise; median higher (175.5 vs 98.8) but medians at n=10 are
  noisy — the grasp is the qualitative difference. Videos
  `v2vid_mini47k5_*.mp4` (14); traj
  `eval_logs/eval_v2_traj_mini47k5.jsonl`. Mini is clean → FULL
  FROZEN EVAL of 047500 submitted: n=60 pick + 20 nav, filmed, tag
  full47k5 — the ledger's headline number.

## v2 HEADLINE — frozen evaluation of the selected checkpoint

- **2026-09-05 — FULL FROZEN EVAL COMPLETE (job 283261, exit 0):
  π₀ fine-tuned on airvla_v2, checkpoint 047500 (T4 winner), v2
  world, validated harness, seed family 97000, tag full47k5, all 80
  episodes filmed.**
  - **PICK n=60: median closest-approach 170.8 mm; 1 picked; 1
    PLACED (episode 9: 9.8 mm approach, weld tick 699, carried to
    the box and released inside, d_bin 29.8 mm — the first complete
    pick-and-place success of the project, on film:
    `v2vid_full47k5_pick09.mp4`).** Near-miss profile: 7/60 episodes
    within 30 mm (9.8✓ 17.8 20.1 24.9 25.3 25.9 27.6), 20/60 within
    ~100 mm. Contact observations (now real): 55/60 episodes fully
    clean; worst offenders ep10 (31 table-tick contacts, 22 object
    strikes) and ep35 (22/3); success ep9 was contact-clean.
  - **NAV n=20: 9/20 success — gate crossing 20/20 (100%), clean
    (0 gate contacts across all 20); failures are all the
    hover-over-target criterion.**
  - **v1 vs v2 (two studies, side by side): median miss 242 →
    170.8 mm (−29%); placed 0/60 → 1/60; v1's 3 'picked' carried
    the latched-lift caveat, v2's single pick is a complete
    task success under a replay-validated gate.** The last-centimeter
    conversion bottleneck is now the sharpest quantified finding:
    11.7% of episodes reach <30 mm but only 1 converts — the
    open-loop-chunk vs every-tick-servo gap (RTC/PAG track is the
    designed answer).
  - Artifacts: videos `Reports/v2_eval_videos/full47k5/` (80),
    success sheet banked; traj `eval_logs/eval_v2_traj_full47k5.jsonl`;
    curve `eval_logs/v2_valcurve_60k.json`; PROV in
    `v2full47k5.log` (script e5ed513b62d5, dep hashes recorded).
  - Selected checkpoint PUBLISHED:
    `hanapasta/airvla_v2_pi0_047500` (private), read-back-verified
    revision `a405a487f891`, 8 files (first upload attempt failed
    401 — job's HF_HOME redirect hid the token; fixed with
    HF_TOKEN_PATH pointing at the standard token file, never read
    or moved).

## E-track — pre-registration (declared 2026-09-05, BEFORE any E run)

Goal (Hana): grasp success ≥60%. Baseline forever: full47k5
(naive-50, 1/60 picked, 1/60 placed, median 170.8 mm). All E
conditions use checkpoint 047500, scene seed 97000, torch seed 1000,
the SAME tick budgets (1200 pick / 500 nav), and the
replay-validated harness — only the declared knob changes. Primary
metric: picked rate on n=60; secondary: placed rate, median miss_mm,
nav success. Each condition gets its own tag; no re-runs under the
same tag; videos on.

- **E1 — execution horizon 10:** execute the first 10 actions of
  each 50-step chunk, then re-infer (replan at 1 Hz instead of
  0.2 Hz). Implemented as `eval_v2.py --exec 10`; `--exec` defaults
  to 50 = EXACTLY the frozen baseline behaviour (flag absent ⇒
  byte-identical code path). Evidence: π0.5-LIBERO 75%→90% from the
  same change. Protocol: one mini (10+4, tag e1mini47k5) to measure
  wall-clock ONLY (no selection on its results), then the full
  n=60+20 (tag e1full47k5) regardless of the mini's numbers.
- **E2 — Real-Time Chunking:** prefix-frozen inpainting during flow
  sampling (Black et al. 2506.07339). Availability of the LeRobot
  implementation on the pinned cluster env to be CHECKED (never
  assumed); if absent, implement prefix guidance in eval-side code
  against the pinned pi0 sampling loop, validated by expert replay
  + a paired mini before any full run.
- **E3 — terminal-corrective flavour + fine-tune:** new collection
  flavour spawning the expert at near-miss hover states (3–10 cm
  offsets around the target, the policy's actual failure
  distribution), ~150 episodes, fine-tune FROM 047500; re-select on
  the existing validation protocol extended with the new episodes'
  val split.
- **E4 — test-time sampling (best-of-N chunks)** and **E5 —
  residual RL fine-tune**: escalation reserves; specified in detail
  only if E1–E3 leave the target unmet (details pre-registered
  before running).

### E1 RESULT (2026-09-05, job 284127, exit 0) — FALSIFIED

- e1mini47k5 (exec-10, 10+4, same scenes): PICK median **337.6 mm
  (naive mini: 175.5), 0 welds**; NAV 1/4 (crossed 4/4). The
  reliable grasp scene (ep9: 9.3–9.8 mm + weld in BOTH naive runs)
  collapsed to 486.6 mm. Verdict: naive 1 Hz replanning is
  HARMFUL for this policy — π₀ draws fresh noise per inference, so
  5× replans inject 5× plan-resampling churn with no cross-chunk
  consistency; the policy dithers between approach plans instead of
  committing (the LIBERO 75→90% transfer does NOT hold here).
- **DOCUMENTED DEVIATION:** the pre-registered "E1 full n=60
  regardless of the mini" is DEFERRED for futility (10–12 GPU-h on
  a decisively harmful arm); Hana can overrule. The mini stands as
  the E1 record.

### E2 pre-registration (declared BEFORE implementation/run)

- **E2 = RTC + exec-10**: LeRobot 0.6's built-in RTCProcessor
  (`RTCConfig(enabled=True, execution_horizon=10)`, LINEAR prefix
  schedule, max_guidance_weight 10 = defaults), `inference_delay=0`
  (synchronous harness — physics pauses during inference; RTC here
  provides cross-chunk prefix consistency, the designed cure for
  E1's churn). Leftover = the previous chunk's unexecuted tail in
  the model's normalized action space, zero-padded to
  max_action_dim (π₀ pads actions with zeros by construction).
  `--rtc` flag; PROV records rtc + guidance params; default-off ⇒
  baseline path untouched.
- **Futility gate (learned from E1):** mini (10+4, tag e2mini47k5)
  first; proceed to full n=60+20 (e2full47k5) ONLY IF the mini
  median < 175.5 mm (the naive mini) OR any weld occurs; otherwise
  stop, record, and escalate to E3 (terminal-corrective data).

### E2 MINI RESULT (2026-09-05, job 284485, exit 0) — GATE PASSED

- e2mini47k5 (RTC + exec-10): PICK n=10 median 250.7 mm, **1 picked
  + 1 PLACED — episode 7: 1.9 mm closest approach (tightest of ANY
  v2 run), weld at tick 381 (≈2× faster than either naive success),
  placed at d_bin 14.0 mm.** Scene 7 was never grasped by any other
  arm (naive ~170 mm, E1 549 mm). Ep8: 34.6 mm where naive managed
  284–303. NAV 3/4 (crossed 4/4) — best of the exec-10 arms.
  Distribution stays bimodal (4 far misses >560 mm); median above
  naive's, but the gate passes on the weld clause. Reading: RTC
  keeps the reactivity of fast replanning while killing E1's plan
  churn — conversion when near improves sharply; far-miss episodes
  are a targeting problem RTC does not address (E3's job).
- **E2 FULL submitted: job 285735** (60+20, tag e2full47k5, filmed,
  h_rt 14 h — RTC guidance adds an autograd pass per denoise step).
  Grasp-moment sheet of ep7 banked; mini videos in
  `v2_eval_videos/e2mini47k5/`.

### E2 FULL RESULT (2026-09-05, job 285735, exit 0)

- e2full47k5 (RTC + exec-10, n=60+20): PICK median 215.8 mm,
  **1 picked + 1 placed — the SAME rate as the naive baseline
  (1/60)**; NAV 6/20 (crossed 20/20; naive 9/20 — fast replanning
  degrades steady hovering). The success (ep24) is the project's
  best-executed grasp: 4.6 mm, weld tick 222 (≈3× faster than
  naive's), placed 28.9 mm. Distribution: near zone DENSIFIED
  (14 episodes <60 mm vs naive's 11) but the sub-30 mm terminal
  cone THINNED (3+1 vs 7) — RTC brings the policy near more often
  and grasps decisively when it converts, yet enters the final
  3 cm less often, and the mini's two conversions did not
  reproduce (marginal conversions flip under replan-compounded
  variance).
- **VERDICT, inference-time rung of the ladder COMPLETE: naive
  1/60 · E1(exec-10) falsified · E2(RTC) 1/60.** Inference-time
  execution changes alone do NOT move the success rate for this
  checkpoint; they redistribute where it fails. The rate must come
  from data (E3) and/or policy improvement (E5). Naive-50 remains
  the best overall configuration (median + nav) with E2 the best
  grasp-execution quality. Success video + traj banked
  (`e2full47k5/v2vid_e2full47k5_pick24.mp4`,
  `eval_v2_traj_e2full47k5.jsonl`).

### E3 pre-registration draft (declared 2026-09-05, BEFORE
### implementation; collection requires Hana's demo sign-off)

- **Motivation (from the full47k5 trajectory diagnosis + E1/E2):**
  failures hover 2–4 cm lateral / 4–13 cm vertical off the target
  for only ~4–5 s then leave; no training episode contains a
  perturbed terminal state or a correction out of one, so the
  policy has never seen "hesitate near the object, then re-align
  and grasp". RTC (E2) improves conversion when near; E3 supplies
  the missing supervision for GETTING from near-miss to grasp.
- **New flavour `terminal` (additive to collect_v2, default-off —
  the collection/eval code paths for existing flavours stay
  byte-identical; harness re-validated by expert replay after the
  edit):** normal flown approach → standoff → deploy → creep
  toward a PERTURBED aim (lateral 2–4 cm uniform direction;
  vertical +3 to +13 cm, ABOVE-ONLY — amended 2026-09-05 before
  implementation: any below-aim hover parks open jaws beside the
  object body with the strike gate armed, so low perturbations
  would mostly burn discarded episodes; above-states cover the
  dominant observed failure modes and correction-from-above
  supervises the same vertical-alignment behaviour) → hesitation
  dwell 10–20 ticks at the perturbed hover → re-target the TRUE
  aim → creep in → fire → weld → stabilize → place. All existing
  safety gates apply; banked only if clean, like every other
  flavour.
- **Plan:** demo (6 eps) for Hana's sign-off → collect ~150
  episodes (seed family 74000) → split extension (episode-level,
  same VAL_FRAC) → fine-tune FROM 047500 (~10k steps, same frozen
  T-protocol otherwise) → validation curve over the new
  checkpoints with the SAME pinned-noise protocol on the extended
  val set → mini gate (beat 175.5 mm or weld) → full n=60+20 under
  naive AND RTC execution (E3 composes with E2).

### E3 amendments (2026-09-05/06, Hana's review + failure taxonomy)

- **Baseline failure taxonomy (all 60 frozen-eval episodes, from
  trajectories + regenerated scenes):** 27 reached the RIGHT
  object's terminal zone (<60 mm lateral; conversion failures), 7
  near-parked (60–150 mm), **16 flew to the WRONG object (12 of
  them into ITS terminal zone, some at 2–8 mm) — language-grounding
  failures**, 10 drifted. 39/60 fly a terminal-zone approach to
  SOME object: control is largely solved; conversion + grounding
  are the two remaining failure families. Confusions are
  asymmetric (10/16 commanded-weight→went-to-penguin — salience +
  nameability bias toward the blue penguin).
- **Terminal flavour REDESIGNED (Hana):** the perturbed-hover and
  lateral-dogleg versions are replaced by drift-then-yaw-correct,
  built only from standard primitives: level nose-first leg to a
  false point (lateral 2–5.5 cm = the MEASURED near-cluster
  distribution, median 32 mm; 6–10 cm short), then the standard
  yaw-in-place re-pointing the gripper at the object, then the
  UNTOUCHED normal creep/fire/grasp (stock creep byte-identical
  for all flavours). Injected misalignment never exceeds the
  measured policy misalignment (Hana: don't teach more drift than
  π exhibits). Status: debugging (local logic check weak; the
  close-range re-yaw swings the 0.2 m jaw lever — being tuned
  before any demo).
- **NEW grounding component — paired-command episodes:** for each
  standard-scene layout, TWO episodes with identical object
  positions/spawn/box side and only the instruction changed
  (runner support: `reset_scene_v2(layout=…)` replay +
  `last_layout`). Rationale: action-only supervision leaves
  instruction-target binding to cross-episode statistics; pairs
  make the sentence the only distinguishing signal.
- **NEW pre-registered secondary metric for ALL future evals:
  grounding rate** — which object's terminal zone (<60 mm lateral)
  the jaws approached: correct / wrong / neither, computed from
  trajectories + regenerated scene positions.

### E3 TRAINING protocol (pre-registered 2026-09-06 ~22:3x, BEFORE
### the merge/training jobs run; overnight chain 286531 → 287020
### (merge+split+push) → 287021 (fine-tune) + 287022 (validation))

- **Data:** `hanapasta/airvla_v21` = aggregate(airvla_v2 600 +
  airvla_v2_e3 ~310, source order preserved). Split: the ORIGINAL
  480/120 verbatim (old episodes never move); terminal 80/20
  episode-level; pairs split AS UNITS (both members one side; a
  split pair would leak its layout); seed 424243.
- **Fine-tune:** FROM 047500 (policy.path init, fresh optimizer),
  batch 4, seed 1000, bf16, grad ckpt, vision unfrozen, steps
  15,000, save every 2,500, cosine schedule COMPRESSED to the run
  (decay_steps 15,000 — documented deviation from the 30k-decay
  recipe so the fine-tune completes a full warmup→decay cycle).
- **Validation:** same pinned-noise paired protocol (seed 31415) on
  the extended val set via the generalized valcurve (repo arg);
  per_flavour now separates old kinds (std/corr/nav) from new
  (term/pairA/pairB) = the forgetting-vs-learning drift detector.
  The UNMODIFIED 047500 is scored on the same windows as
  pseudo-checkpoint 000000 (the baseline row).
- **Selection rule (T4-E3, frozen now):** among fine-tune
  checkpoints, the LOWEST combined val MSE **subject to** the
  original-flavour val MSE (std+corr+nav pooled) not exceeding
  1.10 × the 000000 baseline row. If no checkpoint satisfies the
  constraint, E3 training is judged harmful and 047500 stays the
  policy.
- **Grounding probe (pre-registered):** on val PAIR episodes:
  frame-0 observation, policy queried with BOTH commands (same
  pinned noise), 50-step integrated xy displacement classified
  toward correct/wrong object by cosine against the layout's object
  directions; report correct-heading rate and swap-flip rate for
  the winner AND the 047500 baseline.
- **Eval ladder after selection:** mini gate (10+4, naive-50,
  videos; gate = any weld OR median < 175.5 mm) → full frozen
  n=60+20 naive with grounding rate; an +RTC composition arm may
  follow as its own tagged condition.

### PAG PRECONDITION (recorded 2026-09-07 after Hana's dissertation
### check; flagged in conversation 2026-09-05 but not yet in the
### ledger — her red note caught the gap)

- **The eval platform currently INHERITS the collector's payload
  feed-forward** (V2Platform welds via V2Runner.weld_grasp, whose
  override applies the thrust bookkeeping). Before ANY
  Payload-Aware-Guidance run, eval_v2 needs a `--naive-payload`
  switch routing the weld through the base Runner.weld_grasp (weld
  only, no FF), or PAG's disturbance is absent at test time by
  construction. Impact on banked results: negligible (FF acts only
  while welded; two welds exist across all pure-policy evals; all
  arms share it). For hybrid/E5 FULL evals with frequent carries,
  the FF-on condition must be stated in the results, and any
  PAG-rung comparison must use the naive switch on BOTH sides.

### H1 + E5 pre-registration (2026-09-07, Hana's directive: "we can
### try the hybrid handoff and also the residual RL against dense
### weld")

- **H1 — terminal-servo handoff (hybrid rung):** `eval_v2.py
  --assist R`. When the POLICY brings the jaws within R (horizontal)
  of the commanded object with the arm deployed, the platform's
  scripted terminal servo — the collector's own creep law (3 mm/tick
  frozen-axis creep, +12 mm overshoot, raise-only sag comp,
  measured-rate fire, existing weld gate), 600/600 + 310/310
  validated — flies the final leg; on weld (or a bounded-close /
  300-tick give-back) control returns to the policy for carry and
  place. Engagement is recorded per episode (assisted /
  assist_tick / assist_ticks) and in PROV (assist_r), so hybrid
  numbers are ALWAYS a separate ladder rung, never conflated with
  pure policy. Primary condition R=0.15 on ckpt 047500;
  R-sensitivity (0.10/0.20) as secondaries. V2Platform extracted to
  `platform_v2.py` (shared by eval + validators; assist_r=0 is
  bit-identical to the inline class). Validation gate before any
  GPU eval: replay case A (assist off ⇒ EXACT historical weld ticks
  292/195/241), case B (assist + expert actions ⇒ weld), case C
  (STALLING pilot: expert to 0.20 m then zero actions ⇒ the servo
  alone must complete the weld).
- **E5 — terminal-phase residual RL (learned servo):** the ladder
  mirror of H1 — same activation region, but the last leg is a
  LEARNED residual instead of a script. Design: frozen π₀ (047500)
  serves base actions at the naive cadence; a small residual MLP
  (proprio + sim-privileged jaw-to-target vector; privilege is
  training-time-only and documented) adds bounded per-tick deltas
  (|Δxyz| ≤ 0.01, |Δyaw| ≤ 0.02, |Δgrip| ≤ 0.2). Training episodes
  START at scripted near-miss states (the E3 terminal choreography
  places the drone there) so every sample is decision-relevant and
  episodes are short (~200 ticks ⇒ ~500+ eps/GPU-hour). Reward:
  −jaw-to-aim distance per tick, +weld bonus (episode end), contact
  penalties, small action penalty. Algorithm: compact PPO
  implemented against the pinned env (no new dependencies).
  Deliverables: rl_residual.py (env + PPO), smoke run, overnight
  training, then the SAME gated mini/full ladder with the residual
  active only inside its trained radius.
- **E5 design refinement (2026-09-07, after env smoke):** with the
  zero base the agent is a LEARNED TERMINAL CONTROLLER, not a
  residual — the exact learned mirror of H1's scripted servo. This
  is adopted: training needs NO π₀ inference (pure MuJoCo, CPU-only
  jobs, ~10⁴ episodes/night), and eval integration reuses H1's
  takeover machinery with the learned policy in place of the
  script. Env: `rl_env.py` (expert-delivered standoff starts,
  12-obs proprio+privileged-target, 4-act bounded, dense −distance
  + weld bonus reward; smoke-tested). Ladder rungs stay separable:
  H1 = scripted servo, E5 = learned servo, both policy-triggered.

- **H1C composition arm (pre-registered 2026-09-07, Hana's
  question "why not the plan-C model?"): H1 servo on C-15000** —
  the strongest assemblable system: C's mini put 8/10 episodes
  inside the 0.15 m assist radius (047500: ~5/10), so the servo
  triggers far more often on C's approaches. Kept separate from
  H1-on-047500 so the servo's marginal contribution to the BASELINE
  stays cleanly attributed. Mini = job 296131, tag h1cmini.

### H1C FULL RESULT (2026-09-07, job 298457, exit 0) — BEST SYSTEM
### AT FROZEN n=60

- h1cfull (**C-15000 + terminal servo, R=0.15, n=60+20**): PICK
  median **19.2 mm, 24 picked, 22 PLACED — 36.7% complete task,
  40.0% grasp rate**; NAV 11/20 (crossed 20/20). The mini's 60%
  regressed at scale (n=10 optimism) but H1C remains the best
  measured system on every pick metric. Includes the tightest
  approach ever recorded (ep41: 1.4 mm) and streaks of 5+
  consecutive successes; 2 grasp-but-no-place losses (one with
  carry table-contact) — same category as H1's, review on film.
- **THE FROZEN n=60 LADDER (complete pick-and-place):** pure
  047500 1/60 (1.7%) · pure C-15000 1/60 (best flier: median
  145.8, nav 12/20) · 047500+servo **19/60 (31.7%)** · C-15000+servo
  **22/60 (36.7%)**. Remaining headroom: policy arrival rate
  (~55-60% of episodes never enter the radius), servo heading
  (H1.1), carry/place losses, grounding (~quarter of far misses).

### H1 FULL RESULT (2026-09-07, job 296350, exit 0) — THE FROZEN
### n=60 HYBRID NUMBER

- h1full (**047500 + terminal servo, R=0.15, n=60+20**): PICK
  median **18.2 mm, 21 picked, 19 PLACED — 31.7% complete
  pick-and-place** (35.0% grasp rate) vs the pure baseline's 1/60
  on the identical frozen protocol: a 19× jump at full scale. NAV
  10/20 (crossed 20/20). Welds cluster at ticks ~206–287 with ~50
  servo ticks each (plus one late 736). New category to review on
  film: 2 grasped-but-not-placed episodes (carry/place losses).
  This is the dissertation's hybrid-rung headline pending the
  H1C-full confirmation (job 298457, running next), whose mini
  read 60%.

### H1C MINI RESULT (2026-09-07, job 296131, exit 0) — 60% REACHED

- h1cmini (**C-15000 + terminal servo, R=0.15**): PICK n=10
  **median 8.9 mm, 6 picked, 6 PLACED — 60% complete
  pick-and-place, Hana's target, on the composition arm she
  proposed** ("why not the plan-C model?"). NAV 2/4. One servo
  stall recorded (ep9: engaged, 15.6 mm, no weld — heading again).
  The ladder at mini scale now reads: pure 047500 ~10% picked /
  0-10% placed → pure C 0/10 picked but tightest approaches →
  047500+servo 40% → **C+servo 60%**. n=60 CONFIRMATION dispatched:
  job 298457, tag h1cfull. H1-full (047500+servo) mid-run and
  pacing ~40%.

### C-FULL RESULT (2026-09-07, job 296066, exit 0) — NEW BEST PURE
### POLICY

- e3cfull (C-15000, naive-50, n=60+20): PICK median **145.8 mm**
  (047500 baseline: 170.8), **1 picked + 1 placed** (ep30: 10.2 mm,
  weld tick 285 — expert-fast — placed 13.2 mm); NAV **12/20**
  (047500: 9/20; crossed 20/20). Plan C beats the baseline on
  every metric except success rate (equal), confirming the C-mini
  read at n=60: the terminal-corrective + paired data made a
  better-flying, better-grounded policy whose pure conversion
  remains the bottleneck. C-15000 is now the preferred BASE for
  hybrid composition (H1C mini running).

### H1 MINI RESULT (2026-09-07, job 296054, exit 0) — THE JUMP

- h1mini (047500 + terminal servo, R=0.15): **PICK n=10 median
  20.8 mm, 4 picked, 4 PLACED (40% complete task success — 24× the
  best pure policy)**; NAV 2/4 (assist inert on nav, matches
  047500's own nav). Trigger accounting: servo engaged 6/10,
  converted 4/6, every conversion a full pick-AND-place (74.1 /
  14.2 / 62.8 / 38.0 mm from bin centre; welds at ticks
  279/246/209/736, ~50 assist ticks each). The two servo misses
  (10.2 / 20.8 mm stalls, both penguins) trace to the servo not
  commanding HEADING — penguin grasps need pads-on-head-sides;
  fix identified (slew yaw toward the live-target heading during
  takeover) and reserved as servo-v2 (H1.1) so the pre-registered
  R=0.15 primary runs unmodified first. The 4 non-triggers are the
  familiar far-miss scenes — the policy's arrival rate remains the
  binding factor, which is exactly what the H1C composition arm
  (queued) tests. Videos banked (`v2_eval_videos/h1mini/`).
  **H1 FULL (n=60+20) dispatched: job 296350, tag h1full.**

### C-mini RESULT (2026-09-07, job 292739, exit 0) — GATE PASSED,
### AMENDMENT VINDICATED

- e3cmini (C-15000, naive-50): PICK n=10 **median 109.3 mm** (19.4 /
  49.0 / 53.5 / 75.8 / 108.8 / 109.3 / 129.1 / 143.5 / 411.9 /
  517.9), 0 welds; NAV **3/4** (crossed 4/4); zero contacts in all
  14 episodes. The 1.44× teacher-forced "old-flavour drift" produced
  NO closed-loop degradation — nav BEAT 047500's own mini (2/4) and
  the approach distribution is the tightest of any pure policy
  (7/10 under 145 mm; ep0's 19.4 mm = closest pure-policy approach
  ever on that scene). Mini gate passes on the median clause
  (109.3 < 175.5). Conversion remains zero — the terminal
  behaviour improved approaches but still does not close the last
  centimetres alone. DISPATCHED per the amendment path: full frozen
  eval of C-15000 (job 296066, tag e3cfull) + grounding probe on
  C's checkpoints (job 296067), both queued behind the H1 mini
  (296054).

### E3 RESULTS — collection & first fine-tune (2026-09-06)

- **Collection PERFECT: 310/310 banked in 310 attempts, zero
  rejections** (150 term, 76w/74p; 80 complete pairs; ticks
  345–655, mean 491). Merge: `airvla_v21` = 910 eps / 391,750
  frames (rev 3337e1a5c28d; e3 repo rev 17f06af15c6f); split
  728/182, original verbatim, pairs as units, integrity asserted.
  Manifests/split banked in `eval_logs/`.
- **Fine-tune run A (jobs 287021/287022, full-recipe LR): FAILED
  the pre-registered T4-E3 constraint — 047500 REMAINS THE
  POLICY.** Combined-val curve (baseline 047500 = 0.000062):
  2.5k 0.000187 · 5k 0.000198 · 7.5k 0.000117 · 10k 0.000120 ·
  12.5k 0.000091 · 15k 0.000099. No checkpoint beat baseline; the
  per-flavour splits show GLOBAL churn (old flavours 1.4–3× worse
  AND term/pair flavours above even the baseline's scores) — the
  2.5e-5 peak LR was too aggressive for a 4B fine-tune on a 30%
  data delta; the low-LR tail recovered too late and ticked up at
  15k. Curve banked: `eval_logs/e3_valcurve.json`.
- **Plan B (recipe amendment, staged BEFORE the verdict, submitted
  on it): fine-tune run B = identical except peak LR 5e-6 (5×
  gentler), decay to 5e-7 over 10,000 = steps, output
  pi0_e3b_out.** Jobs 290276 (train) + 290277 (validation daemon,
  same protocol, out e3b_valcurve.json). Same selection rule.
- **Grounding probe submitted on run A's checkpoints anyway (job
  290275)**: if even the churned fine-tune shows a flip-rate rise
  over baseline, the paired-command mechanism is confirmed
  independent of the LR mishap. (First attempt crashed on a real
  probe bug — parquet rows carry task_index, not the task string —
  fixed and resubmitted as 290283.)
- **Fine-tune run B (jobs 290276/290277, peak 5e-6): FAILED
  selection — the opposite failure to run A.** Curve: 2.5k
  0.000070 · 5k 0.000073 · 7.5k 0.000075 · 10k 0.000078 (baseline
  0.000062). No churn (old-pooled only 5–15% over the ≤5.0e-5
  constraint) but ALSO no learning — the term flavour never
  improved past the baseline's own 1.5e-4. Diagnosis: EXPOSURE —
  at batch 4 × 10k steps the model samples ~10% of the combined
  frames, seeing each new-flavour frame ~0.3× in expectation.
  Curve banked (`e3b_valcurve.json`).
- **Plan C RESULT (2026-09-07 morning): first run to LEARN — and
  a DOCUMENTED AMENDMENT (approved by Hana).** Curve (baseline
  0.000062): 2.5k 0.000118 → 5k 0.000117 → 7.5k 0.000081 → 10k
  0.000081 → 12.5k 0.000077 → 15k 0.000077 → 17.5k 0.000078. The
  terminal flavour dropped BELOW baseline from 10k on (1.43–1.46e-4
  vs 1.49e-4 — the first genuine new-behaviour learning in any
  run), but old-flavour drift plateaued at ~1.44× baseline
  (6.55e-5 vs the ≤5.0e-5 limit) ⇒ fails the strict pre-registered
  constraint. AMENDMENT (Hana, "look at B then maybe circle back"):
  the constraint is a forgetting PROXY and this project has
  repeatedly measured that teacher-forced magnitudes do not predict
  closed-loop behaviour — so C-015000 (lowest drift + learned term)
  goes to the pre-registered MINI GATE, letting closed-loop reality
  adjudicate. Mini = job 292739 (10+4, filmed, tag e3cmini); the
  weld-or-median gate still protects the full eval. Strict-rule
  outcome recorded regardless.
- **Plan C submitted (jobs 291164 train / 291165 val)**: fresh
  fine-tune from 047500 with the training LIST rebalanced — all
  248 new-flavour training episodes + a stratified HALF of the
  original ones (~50% of each batch is new behaviour) — 20k steps
  at the proven-gentle peak 5e-6 ⇒ ≈10× run B's new-frame
  exposure, using no unverified sampler mechanics (episode-list
  rebalancing only). Same selection rule; note the old-flavour
  drift constraint is now the live risk (halved old exposure), so
  the 1.10× guard is doing real work.
- **2026-09-04 15:08 — PROTOCOL AMENDMENT (Hana): training EXTENDED
  30k → 60k** because the validation curve is still descending at 25k
  (0.000217 → 0.000061 from 10k to 25k with no sustained upturn — the
  17.5k bump was noise). T4 selection rule UNCHANGED (lowest val
  action-MSE), now applied over the full 60k curve; same pinned noise
  keeps every new point exactly paired with the old curve. Jobs:
  280833 resumes from checkpoint 030000 (`--resume=true
  --steps=60000`, janitor active, log v2train60k.log); 280834 is a
  rolling-validation daemon that waits for sweep 276984 to finish,
  then scores each new checkpoint as it appears (valcurve_v2.py
  resumable passes, log v2val60k.log). Full n=60 eval waits for the
  60k curve's winner.
