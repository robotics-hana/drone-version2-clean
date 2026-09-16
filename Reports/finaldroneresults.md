# Final drone results — v2 campaign tracker

Started 2026-09-01. This file is the single tracker for the v2 dataset →
training → evaluation campaign: pre-registered definitions first (locked
BEFORE collection/training), then results as bullets as they land.
Nothing in the pre-registration section may be edited after training
starts — corrections get dated addenda.

## COMPLETED-EXPERIMENT INDEX (what / why / headline result)

Digest of every completed v2-era experiment — the question each was
built to answer and its decisive number. Details, pre-registrations,
amendments and provenance live in the dated sections below.

| Experiment | Why (the question) | Headline result |
|---|---|---|
| v2 base (047500) | Does π₀ learn the task from 480 clean demos? | Approach yes, grasp no: 1/60 picks, 170.8 mm median, 42/60 flew-to-target, 9/20 nav |
| E1 exec-10 | Does 5× faster replanning fix the terminal? | FALSIFIED — plan-resampling churn doubled the miss (337.6 mm mini); LIBERO transfer does not hold |
| E2 RTC | Does prefix-consistent replanning fix it? | Cures churn, not the rate: 1/60 (= naive), best-executed grasp (4.6 mm) but enters final 3 cm less often; nav degrades 9→6 |
| F1/C terminal-correctives + 80 pairs | Does failure-matched data help? | Precision yes, rate no: 145.8 mm (170.8), wrong-target 18-19→13-14, grasp still 1/60 |
| F2/D pair scaling (80→304) | Does 3.8× paired data improve grounding? | OVERDOSE — wrong-target worsens to 20/60, nav 12→10, precision 204.9 mm; dose-response is U-shaped |
| D-KI knowledge insulation | Is D's regression backbone erosion or data? | BOTH, apportioned: frozen backbone recovers precision (88.2 mm true, best pure) + nav (12/20) but only half the grounding (43 vs 47) |
| H1 scripted servo | Is the last 150 mm solvable by a scripted terminal controller? | Yes: 21-24/60 grasped (vs 1), 31.7-36.7% placed; conversion 62% — heading brittleness |
| E5 learned servo (DAgger) | Can a learned closer beat its scripted teacher? | Yes — student surpasses teacher: **E5C 36/60 grasp (60%), 27/60 placed (45%) = BEST SYSTEM**; conversion 90% |
| E5D / E5-DKI compositions | Which policy pairs best with the servo? | C wins: E5D 35%, E5-DKI 43.3% (best-ever true median 9.3 mm; loses on conversion 85% vs 90%) |
| PAG feed-forward ablation | Does the loaded-thrust feed-forward do anything? | INERT — mm-identical with it removed (orphaned by MPPI→PD migration); active arm gravity-moment FF is separate and load-bearing |
| ACT / DP baselines | Do from-scratch policies match the VLA? | No: chance-level grounding (28, 35/60), 0-1 picks, medians 249-313 mm |
| Target-true metrics | Are medians confounded by wrong-object flights? | Yes — general vs target-true median separates selection from precision for every run (retro-computed for all) |
| Solo probe (4 arms, n=30) | Is the deficit target selection or approach skill? | Dissociates: C 97% flew-to-target solo (78% paired) = selection; ACT 53%/DP 63% solo ≈ paired = approach incompetence |
| Paraphrase OOD (E5C, n=60) | Is grounding template memorization? | No — COMPLETE PARITY under 5 unseen phrasings (36/28/46 vs 36/27/45); wording brittleness falsified |
| Latency (GB10) | Is the system real-time viable? | 235.9 ms median/inference, 4.72 ms/tick amortized at exec-50 — yes |
| V3-arm (relabel + train) | Should the VLA control the arm? | **NO — measured**: v3full 38/60 flew, 139.4 mm true, 264.2 mm general, 7/20 nav, 0 picks — worse than 047500 on EVERY axis; the arm-as-platform design is vindicated by a one-variable experiment |
| DAgger/FT-DAG (approved 2026-09-14) | Does on-policy corrective data fix the pure-VLA terminal? | Trial: mechanism validated (4/4 banked, 3 near-seams 181-213 mm); full collection in flight |

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
- **E5 AMENDMENTS (2026-09-07/08, three defects found and fixed
  before any result was read):**
  1. *Zero-weld run misdiagnosed:* PPO ran 1,092 episodes with 0
     welds; first read as a hard-exploration cliff and a BC warm
     start was added (job 299255→299529). Post-mortem falsified the
     cliff story: `rl_env`'s zero base held the ABSOLUTE grip
     channel at 1.0 (open) and the |Δgrip| ≤ 0.2 residual clamps at
     0.8 — the weld was **unreachable by construction**, not
     hard to explore. Fix: base grip = current grip (Δgrip becomes
     a rate; closed reachable in ~5 ticks). Job 299529 killed —
     trained against the broken env.
  2. *Verbatim servo clone diverges closed-loop:* BC on
     `_servo_tick` hit MSE 0.0033 / cos 0.94 to the teacher yet
     welded 0/8 closed-loop, drifting to 330–650 mm. Cause: the
     servo's frozen approach axis (a_ufix) + forward ratchet
     (a_fwd) are internal latches NOT in the 12-dim obs — the
     teacher is not a function of the observation, so the clone
     regresses the unresolvable x-component to a smeared mean
     (measured mean |err|: x 0.106 vs y 0.002; ~1 mm/tick lateral
     bias, compounding). Fix: **Markovian teacher** in rl_bc.py —
     same law, axis re-derived from the CURRENT bearing each tick
     (body-heading fallback inside 8 mm where bearing is unstable),
     no ratchet. Validated 6/6 welds in TerminalEnv (t=115–148)
     before adoption. platform_v2.py untouched (frozen-eval dep).
  3. *rl_bc imported rl_train, executing the whole PPO trainer at
     import* (no __main__ guard): shared classes extracted to
     rl_nets.py.
  Gate before resubmission: local 20-episode Markov-teacher BC +
  closed-loop weld test at std 0.0 and 0.22 (PPO's init noise).
- **E5 GATE CAMPAIGN (2026-09-08, five local gates before the chain
  was allowed back on the cluster; each failure diagnosed on
  instrumented traces, not theory):**
  - *Gate 2 (Markov teacher alone): 0/8.* Still-unobservable
    inputs: the SETPOINT the creep law steers from, the
    object-specific grasp params, and world-frame components baking
    the approach heading into the data → 18-dim BODY-frame obs
    (aim-jaw, sp-jaw, vel body; angvel; ap/yerr/grip; ap_lo/ap_hi/
    close) + DART noise injection (teacher-under-noise weld 1.00).
  - *Gate 3 (obs+DART): 0/8 — but the trace flipped the story.* The
    clone tracks the teacher nearly perfectly and closes 379→42 mm;
    it then NEVER STOPS (stop/fire is ~2-5% of ticks; uniform and
    even 20×-weighted MSE both learn "creep always") and flies
    through the object at 3 mm/tick — the 256-800 mm "divergence"
    distances are exactly 300-tick fly-through, not instability.
  - *Gate 4 (DAgger, 3 rounds): 0/4, poisoned labels.* Rolling the
    clone and labeling with the teacher flooded the dataset with
    fire labels (41 → 5,817 of 9,178): the teacher's fire latch
    (a_fired) latches on the clone's fly-past and labels every
    subsequent far-away tick "close", where a closed-empty gripper
    can weld nothing and no reopen label exists.
  - *Gate 5 (state-inferred latch + reopen recovery): the real
    result.* Fire latch inferred from the grip (in the obs);
    closed-empty-far states labeled REOPEN (a recovery the scripted
    teacher never needs, a learner recovering from its own miss
    does). Weld count still 0/4 — but the instrumented eval shows
    the clone now APPROACHES, STOPS, AND HOVERS AT THE OBJECT:
    dmin 5/6/30/33 mm, end-hover 14-103 mm, no fly-through. Only
    the decisive grip close is missing (drifts 1.00→0.82; weld
    needs ~0.5).
  - **Reframed go/no-go (documented deviation):** the BC stage's
    job is DELIVERY INTO THE WELD BALL, not welding — the decisive
    close is a 1-D discovery PPO makes from dense hover states
    (±0.045/tick grip noise over 100+ in-ball ticks, +10 weld
    bonus). Chain resubmitted as job 301570 (DAgger BC 120+3×60 →
    PPO 9 h from the best round's actor).
  - **RESULT (2026-09-08, job 301570): at cluster scale the clone
    welds by pure imitation.** DAgger rounds 0/6 → 0/6 → **6/6 →
    6/6** (56,559 pairs; ~15× the local gates' data — scale was the
    missing ingredient, the reframe wasn't even needed). PPO from
    the best actor opened at **weld100 = 1.00 on its first
    iteration** (27/27 episodes welding UNDER full sampling noise,
    std 0.22) — the 9 h PPO leg now optimizes speed/consistency
    from ceiling rather than discovering the weld. The E5 ladder
    rung (learned terminal servo vs H1's scripted one) is
    functionally established pending its gated mini/full eval.
  - **PPO CHURN-KILL + ACTOR SELECTION (2026-09-08, rule pre-stated
    at the first sub-0.90 alarm, executed on trigger):** weld100
    held 0.95–1.00 through iteration ~60 while episodes shortened
    (PPO learning SPEED under the time-pressure term), then hit the
    kill rule at iterations 70–71 (0.88 → 0.83, two consecutive
    sub-0.90 reads, value loss 1.6 → 5.2 — the churn signature; PPO
    hyperparameters are pre-registered so no mid-run retuning). Job
    killed at ~iteration 71 / 3,373 episodes. Selection among
    preserved actors by DETERMINISTIC closed-loop weld rate over 6
    shared-seed episodes (seed 58000), tie-break mean time-to-weld:
    bc_init 5/6 @130t · it0040 **6/6 @99t** · it0050 5/6 @68t ·
    it0060 5/6 @66t (later actors traded welds for speed — the
    churn tipping over). **FROZEN E5 ACTOR = actor_it0040 →
    e5_actor_final.pt** (cluster md5 b005267b4932b5895eb4d39aaad2
    2654, repo git-blob 4569c3ad1751fe103aa534bc5c761d504e3dd89e,
    also in-repo at Sim'n'Real/Mujoco/e5_actor/). PPO's net
    contribution over BC: +1 weld and 24% faster (130 → 99 ticks).
    Next: E5 gated mini — the learned actor in H1's takeover
    machinery in place of _servo_tick (new flag ⇒ new freeze).
  - **E5 EVAL INTEGRATION (2026-09-08):** platform_e5.py
    (V2PlatformLearned: ONLY _servo_tick overridden; obs via
    rl_env.obs_vec — the training env's own function, parity by
    construction; action application mirrors rl_env.step verbatim)
    + eval_v2 `--learned-servo actor.pt` (requires --assist; PROV
    records actor sha; flag absent ⇒ no new imports, all paths
    untouched). replay_e5_validation.py PASS 4/4: case A exact
    banked weld ticks 292/195/241 (inert with assist off), B/C the
    LEARNED servo welds ~20 ticks FASTER than the expert's own
    creep (274/175/220), N no engagement outside radius.
  - **E5C MINI #1 (2026-09-08, job 305176, tag e5cmini, exit 0) —
    DEPLOYMENT-CONTRACT BUG, DIAGNOSED AND FIXED:** n=10 median
    17.8 mm, 2 picked, 1 placed (vs h1cmini's 6/6). Engagements
    8/10 — identical delivery to h1cmini (paired scenes) — but
    conversions 2/8. Every failure shares one signature: miss
    5–29 mm with 65–71 assist ticks: the platform's
    45-ticks-after-fire give-back clock, keyed on the grip crossing
    0.8, fired during the learned actor's gradual grip EASING
    (scripted servo snap-closes; the actor was trained under the
    env's plain 300-tick episode with no fire clock) and yanked
    control back mid-grasp — ep08 was pulled off at 5.2 mm. Fix:
    learned-servo give-back = the 300-tick cap only (the training
    contract). Validation re-PASS 4/4. Re-run = job 305571, tag
    e5cmini2. (Nav 1/4 vs h1cmini's 2/4 is the documented cross-run
    GPU jitter — nav never touches the servo.)
  - **E5C MINI #2 (2026-09-08, job 305571, tag e5cmini2, exit 0) —
    LEARNED SERVO MATCHES SCRIPTED ON THE MINI:** n=10 median
    15.3 mm, **6 picked, 5 placed** (h1cmini scripted: 6/6, same
    paired scenes); NAV 3/4. The contract fix moved conversions
    2/10 → 6/10. Character difference: the learned servo is slower
    (two grasps at 140–152 assist ticks vs the script's ~30–60 —
    trained for reliability within the 300-tick budget), and one
    weld (ep8, 30-tick engage) was lost during carry. MINI GATE
    PASSED → full n=60+20 dispatched: job 305965, tag e5cfull.

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
  consecutive successes; 2 grasp-but-no-place losses, both plush
  penguin, DIAGNOSED ON FILM (contact sheets in
  Reports/v2_eval_videos/h1cfull/): ep02 = late arrival (policy
  wandered ~900 ticks; servo grasped cleanly at weld_tick 936 of
  1200; episode timed out mid-carry 1.2 m from bin — a time-budget
  loss, not a drop) · ep46 = low carry (weld_tick 487, then 45
  table hits dragging the plush along the table until it was lost
  0.76 m from bin — a carry-altitude loss). Neither is a servo
  grasp failure; fixes live in carry height and arrival speed,
  not the terminal controller.
### THE FROZEN n=60 LADDER (canonical table — all four systems,
### identical frozen protocol: seed family 97000, paired scenes,
### v2 world, validated harness, n=60 pick + 20 nav)

| System (frozen n=60) | Job, tag | Grasped (picked) | Placed (complete task) | Median approach | Nav |
|---|---|---|---|---|---|
| Pure π₀ 047500 | 283261, full47k5 | 1/60 (1.7%) | 1/60 (1.7%) | 170.8 mm | 9/20 |
| Pure Plan-C fine-tune (C-15000) | 296066, e3cfull | 1/60 (1.7%) | 1/60 (1.7%) | 145.8 mm | 12/20 |
| Hybrid: 047500 + terminal servo (R=0.15) | 296350, h1full | 21/60 (35.0%) | 19/60 (31.7%) | 18.2 mm | 10/20 |
| Hybrid: Plan-C + scripted servo (H1C) | 298457, h1cfull | 24/60 (40.0%) | 22/60 (36.7%) | 19.2 mm | 11/20 |
| **Hybrid: Plan-C + LEARNED servo (E5C)** | **305965, e5cfull** | **36/60 (60.0%)** | **27/60 (45.0%)** | **12.8 mm** | **11/20** |
| Pure Plan-D (overdose arm) | 326215, dfull | 1/60 (1.7%) | 1/60 (1.7%) | 204.9 mm | 10/20 |
| Hybrid: Plan-D + learned servo | 326216, e5dfull | 28/60 (46.7%) | 21/60 (35.0%) | 16.4 mm | 11/20 |
| ACT baseline (language-blind) | 321544, actfull | 1/60 (1.7%) | 1/60 (1.7%) | 313.5 mm | 10/20 |
| Diffusion Policy baseline (language-blind) | 324360, dpfull | 0/60 | 0/60 | 249.8 mm | 0/20 |

  Remaining headroom (corrected 2026-09-08, per-episode analysis
  of h1cfull): **wrong-object flights 14/60** (miss in the
  300–700 mm separation band, 9/14 on "weight" commands —
  grounding; the servo never triggers at the wrong object) ·
  drift 7/60 (150–300 mm) · **servo conversion 24/39 engagements
  (62%)** — the engaged-but-not-grasped remainder is the servo's
  heading-stall weakness (H1.1) · carry/place losses 2 (time
  budget + carry altitude, film diagnosis above). Hana's video
  observation (drone heading to the wrong target) is the first
  item, quantified.

### PLAN D PRE-REGISTRATION (2026-09-08, Hana's approval: "we can
### do it" + hybrid arm confirmed + "check the learning rate")

- **Hypothesis:** scaling the paired-command grounding contrast
  (E3's 80 pairs → 300) reduces wrong-target flights (currently
  13–14/60 for Plan C) and thereby lifts the hybrid rate.
- **Collection:** pairs ONLY, 300 units (600 eps), seed 75000 (next
  in the 7x000 family), existing collecte3 machinery with n_term=0.
  Job 305829, repo hanapasta/airvla_v2_d.
- **Merge:** airvla_v21 + airvla_v2_d → airvla_v22; v21 split
  VERBATIM, D pairs split as units, split seed 424245
  (v2_merge_d.py).
- **Training:** from C-15000, PLAN C'S RECIPE VERBATIM (peak 5e-6,
  cosine → 5e-7, 20,000 steps, batch 4, seed 1000) — the LR is
  deliberately NOT retuned (A churned at 2.5e-5, B under-learned, C
  passed on this schedule; identical recipe ⇒ any grounding
  movement is attributable to the data). Episode list = C's own
  list rebuilt deterministically (seed 424244) + all D train pairs
  (d_train_wrapper.py, output pi0_d_out).
- **Gates (unchanged machinery):** validation curve every 2,500
  steps with flavour split; old-flavour churn alarm at 1.10×;
  closed-loop mini (n=10, beat C's 145.8 mm median or weld) before
  any full eval.
- **Frozen evals on pass:** pure D (attribution: wrong-target count
  vs C's 13–14) and D + scripted servo (headline), n=60+20, seed
  family 97000, paired scenes as all prior rungs. Existing rungs
  are NOT re-run — new rows only.
- **AS-BUILT DATA RECORD (final banked counts, vs the 300-pair
  plan above):** **224 paired units = 448 episodes**, collected
  across three time-budgeted jobs after the wall-kill loss —
  airvla_v2_d2 (seed 76000) 270 eps/135 pairs · airvla_v2_d3
  (77000) 90 eps/45 pairs · airvla_v2_d4 (78000) 88 eps/44 pairs.
  A fourth job (d5, seed 79000) was dropped by the stated
  slow-node rule at 3 episodes. Merged with airvla_v21 (910) →
  **airvla_v22 = 1,358 episodes**, pushed to HF and read-back
  verified (revision 85978f65). Extended split **1,086 train /
  272 val**, the v21 split verbatim plus the 224 new pair-units
  split 179/45 as units. Total paired dose in training:
  **304 pairs** (E3's 80 + D's 224) = 3.8× the E3 dose.
- **AS-BUILT TRAINING RECORD:** job 321543, **20,000 steps in
  8 h 47 min, exit 0**, loss ~0.042–0.056 over the final decade;
  8 checkpoints. Episode list **846 = 488 C-list + 358 D-pair**
  episodes (42% new). TWO FALSE STARTS preceded it, both
  root-caused and structurally closed: (1) job 317458 —
  lerobot refuses a pre-existing `output_dir`, and the earlier
  killed attempt had left one (fix: the job now `rm -rf`s its
  output dir before launching); (2) job 321088 — **`Disk quota
  exceeded (os error 122)` at the first checkpoint save**, with
  ~380 GB reclaimed by a janitor pass (all finished runs'
  `training_state` optimizer dirs, the dead A/B fine-tune models,
  hub-backed dataset caches, the torn d-repo) while keeping
  047500, 060000, C-15000, the E5 actor and airvla_v22. Neither
  crash cost data, only ~2 h of wall clock.
- **AS-BUILT VALIDATION CURVE** (extended val set, 272 episodes,
  flavour split; baseline = unmodified C-15000 scored on the same
  windows as pseudo-checkpoint 000000; job 323960):

| Checkpoint | Combined MSE | Old-flavour ratio vs baseline |
|---|---|---|
| baseline C-15000 | 7.733e-5 | 1.000 |
| 002500 | 7.535e-5 | 0.947 |
| 005000 | 6.867e-5 | 0.918 |
| 007500 | 8.789e-5 | 1.070 |
| 010000 | 6.958e-5 | 0.996 |
| 012500 | 7.320e-5 | 0.939 |
| **015000 ← SELECTED** | **6.797e-5** | **0.881** |
| 017500 | 7.038e-5 | 0.913 |
| 020000 | 7.195e-5 | 0.925 |

  Every checkpoint passed the 1.10× churn bound and seven of
  eight scored BELOW the baseline with old-flavour rows improved
  — offline, Plan D looked like the campaign's best fine-tune.
  That is precisely what makes its closed-loop regression the
  sharpest offline/online divergence on record here.
- **SELECTION + GATE RECORD (2026-09-12, overnight):** curve
  complete, all 8 checkpoints under the churn bound; **D-15000
  selected** (mse 6.80e-5 < baseline 7.73e-5; old-flavour ratio
  0.881 — the pairs IMPROVED old behaviours). Mini (job 325626):
  median 146.4, 0 welds, flew-to-target 8/10, nav 3/4.
  **DOCUMENTED AMENDMENT:** the mini gate ("beat 145.8 or weld")
  was missed by 0.6 mm (0.4%) — far inside the campaign's measured
  mini↔full noise (C's own mini→full swing: 109.3→145.8) — while
  every secondary indicator was green (val better than baseline,
  old flavours improved, selection 8/10, nav 3/4). Following the
  C-gate precedent, the fulls proceed with this note rather than
  treating sampling noise as a verdict. The headline eval was
  amended from D+scripted to **D+LEARNED servo** (the learned
  servo superseded the scripted one as best system, e5cfull).
  Dispatched: dfull = job 326215, e5dfull = job 326216.
- **D FULL RESULT (2026-09-12, job 326215, exit 0) — HYPOTHESIS
  FALSIFIED AT PURE-POLICY LEVEL:** PICK median **204.9 mm, 1
  picked, 1 placed**; TARGET-TRUE **flew-to-target 40/60** (C:
  47/60 — a REGRESSION), true median 110.7 (C: 87.6); NAV 10/20
  (C: 12/20). The 4× paired dose did not improve closed-loop
  grounding; it worsened every axis, despite BETTER validation
  MSE (6.80e-5 vs 7.73e-5) with improved old-flavour rows — the
  campaign's starkest offline/online divergence yet, and in
  hindsight the mini's 0.6 mm gate miss was signal, not noise.
  Dose-response is now non-monotonic: 0→80 pairs helped
  (wrong-target 18→13), +224 more hurt (→20/60). Candidate
  mechanisms (not adjudicated): pairs-only new data at 42% of the
  list diluted terminal-corrective exposure; pair scenes
  over-represented relative to the eval distribution. The honest
  reading for the dissertation: paired-command grounding has a
  useful dose and an overdose at this data scale.
- **E5D FULL RESULT (2026-09-12, job 326216, exit 0) — THE
  CAMPAIGN'S LAST RUNG:** D-15000 + learned servo: PICK median
  16.4 mm, **28 picked (46.7%), 21 placed (35.0%)**; TARGET-TRUE
  flew-to-target 39/60, true median 10.5 mm; NAV 11/20. Below
  C+learned servo (60% / 45%) — the composition faithfully
  transmits pure D's grounding regression (arrivals 39 vs 45).
  **FINAL CAMPAIGN VERDICT: Plan C + learned terminal servo is
  the best system — 60% grasp, 45% complete pick-and-place — and
  Plan D closes as the measured overdose arm of the dose-response
  curve (0 pairs: 18-19 wrong-target · 80: 13-14 · ~300: 20).**

### D-KI PRE-REGISTRATION (2026-09-12, Hana's approval: "lets try
### knowledge insulation ... do a mini trial to check for errors
### before committing to full run")

- **Hypothesis:** Plan D's grounding regression is BACKBONE EROSION
  under full fine-tuning — action-expert gradients degrading the
  VLM's language representations (Driess et al., arXiv:2505.23705,
  the π₀ authors' Knowledge Insulation result; the whole campaign
  trained with train_expert_only=false).
- **Design:** identical to Plan D in every respect (airvla_v22,
  same episode list, 20k steps, same LR schedule, seed 1000)
  except `freeze_vision_encoder=true` + `train_expert_only=true`.
  One-flag causal test. dki_train_wrapper.py.
- **Gates:** 200-step smoke first (Hana's mini-first rule), then
  full train → val curve (same machinery, baseline C-15000) →
  closed-loop mini → fulls only if the mini shows grounding
  recovery. **Primary endpoint: flew-to-target vs C's 47/60 and
  D's 40/60.**
- **Interpretation, stated in advance:** recovery to ≥47/60
  confirms erosion and turns the pair-scaling result positive;
  no recovery strengthens the overdose finding against its main
  objection. Cluster: Myriad (C-15000 checkpoint locality).
- **SMOKE GATES PASSED (2026-09-12, both clusters):** Myriad job
  329427 — 200 steps, **exit 0**, loss 0.065, checkpoint saved:
  the KI flags are accepted by the trainer with no config
  incompatibility. Insulated training runs at **0.83 s/step vs
  the full fine-tune's 1.2 s/step** (gradients only through the
  action expert), so the 20k run costs ~4.6 h instead of 8.8.
  A parallel config-smoke on Sparks (run during a Myriad network
  outage, using 047500 + airvla_v21 since C-15000/v22 are
  Myriad-local) also completed cleanly at **3.45 s/step**,
  confirming π₀ training is ~4× slower on the GB10 and that
  Myriad routing was correct. **Full D-KI training = job 330045.**
- **D-KI TRAINING + CURVE (2026-09-13, job 330045 exit 0, curve
  job 330291):** 20,000 steps completed; validation curve
  complete, all 8 checkpoints inside the churn bound (max 1.023).
  **Selected: D-KI-005000** (mse 7.2646e-5, old-flavour ratio
  0.953) by the same pre-registered rule.

| Checkpoint | **D-KI** MSE | old-ratio | Plan D MSE (full FT) |
|---|---|---|---|
| baseline C-15000 | 7.7331e-5 | 1.000 | 7.7331e-5 |
| 002500 | 7.8698e-5 | 1.011 | 7.5348e-5 |
| **005000 ← SELECTED** | **7.2646e-5** | **0.953** | 6.8665e-5 |
| 007500 | 7.7655e-5 | 1.009 | 8.7888e-5 |
| 010000 | 7.6070e-5 | 0.993 | 6.9584e-5 |
| 012500 | 7.7581e-5 | 1.023 | 7.3197e-5 |
| 015000 | 7.6880e-5 | 1.009 | 6.7972e-5 |
| 017500 | 7.6542e-5 | 1.007 | 7.0375e-5 |
| 020000 | 7.7244e-5 | 1.014 | 7.1948e-5 |

- **THE OFFLINE METRIC NOW POINTS AGAINST THE HYPOTHESIS — BY
  DESIGN.** Insulated training fits the data notably WORSE than
  full fine-tuning at every checkpoint (best 7.26e-5 vs Plan D's
  6.80e-5, and barely under the untouched baseline's 7.73e-5),
  which is exactly what freezing the backbone should do: less
  capacity to absorb the new data. The hypothesis predicts the
  OPPOSITE ordering closed-loop. This makes D-KI a genuinely
  strong test: if its grounding recovers toward C's 47/60 while
  its validation MSE is worse than D's, the offline/online
  divergence is confirmed twice over and in opposite directions.
- **D-KI FULL RESULT — PRIMARY ENDPOINT (2026-09-13, job 331748,
  picks complete, navs finishing):** n=60 pick: **flew-to-target
  43/60** (D: 40, C: 47), **target-true median 88.2 mm** (D:
  110.7, C: 87.6 — precision FULLY recovered), general median
  **117.8 mm — the best pure policy of the campaign** (C: 145.8),
  0 picked; **NAV 12/20 — matching C's campaign best** (D had
  regressed it to 10/20); exit 0. **Interpretation per the pre-registered endpoints:
  PARTIAL grounding recovery — Plan D's regression decomposes
  into BOTH candidate mechanisms.** Backbone erosion (fixed by
  insulation) accounts for all of the precision damage and about
  half the grounding deficit; a residual pair-overdose effect
  (43 vs 47, surviving a frozen backbone) accounts for the rest.
  The offline/online divergence is confirmed in both directions
  as designed: D-KI fit validation WORST of the line (7.26e-5 vs
  D's 6.80e-5) yet flies with the line's best closed-loop
  precision. Servo-composition arm: gate read strictly (≥47) not
  met; decision on running it referred to Hana with a
  recommendation to proceed given the precision recovery.
- **MINI-GATE POWER NOTE (stated before the result):** at n=10 the
  mini CANNOT discriminate 47/60 (78%) from 40/60 (67%) — both C
  and D scored 8/10 flew-to-target at mini scale. The D-KI mini
  (job 331457) therefore serves as a crash/sanity gate only; the
  **primary endpoint is the n=60 full**, which proceeds unless the
  mini is catastrophic (e.g. flight breakdown or ≤4/10
  flew-to-target).
- **OPERATIONAL NOTE (2026-09-12):** Myriad was unreachable for
  several hours (TCP timeouts to the login node; DNS fine, Sparks
  unaffected — a UCL-side incident, not the scheduled work).
  UCL's calendar shows a **planned Myriad outage 24–25 September
  2026** (central switch replacement, no access); any remaining
  cluster work must land before it.

### V3-ARM PRE-REGISTRATION (2026-09-13, Hana: "train a v3 vla,
### same training demonstrations as v2, but have the policy
### control the arm as well")

- **Question:** does giving the VLA arm-joint control help or hurt,
  at this data scale? (The design discussion predicts: the
  demonstrated arm signal is near-deterministic given phase, so
  supervision adds little information and two dims of flow noise;
  measured slew/stability limits argue for platform enforcement.
  Either outcome is the measured answer to a real design question.)
- **Data — NO recollection:** relabel airvla_v2's padded action
  dims 3,4 with per-step arm-joint deltas reconstructed from
  consecutive proprio states (episode-boundary-safe; last frame 0),
  action stats refreshed → **hanapasta/airvla_v3** (relabel_v3.py).
  Same 600 demonstrations, same 7-dim layout, byte-identical
  everything else.
- **Training:** the v2 baseline's exact from-base recipe (π₀ base,
  30k steps, batch 4, seed 1000, full fine-tune) so **v3-arm vs
  047500 is a one-variable comparison: arm supervision only.**
- **Eval:** new `--policy-arm` flag — platform applies action dims
  3,4 as arm-joint deltas, clipped per tick to the flight-validated
  slew bound (0.06/tick), REPLACING the phase-based q_travel/
  q_carry switching; deploy flag retained for metrics only. New
  flag ⇒ compile + expert-replay validation before use (replay
  with relabeled expert actions must still weld).
- **Gates:** relabel --dry verify → training smoke (200 steps) →
  full train → val curve (T4-style selection) → closed-loop mini →
  frozen full n=60+20. Mini-first at every stage.
- **RELABEL DONE (2026-09-13, job 332663, V3RELABEL-EXIT=0):**
  patched 239,520 frames; **|d_arm| mean 0.0019, p99 = max =
  0.0300 rad/step** — the hard ceiling at exactly 0.03 is the
  platform's slew-rate cap showing through the reconstruction,
  precisely the near-deterministic signature the design argument
  predicted. Refreshed action std dims 3,4 ≈ 0.0073 (same order as
  the position dims → healthy normalization). Verified nonzero on
  reload; pushed **hanapasta/airvla_v3**; local copy at
  `~/Scratch/hf_cache/lerobot/hanapasta/airvla_v3` (the offline
  path training reads).
- **TRAINING SMOKE SUBMITTED (job 332683, v3smoke.job):** built by
  sed-transforming v2train.job on the cluster (no whole-file scp);
  word-level diff audit of the train command shows exactly 4
  changes — dataset→airvla_v3, output→pi0_v3_smoke, steps 30000→200,
  save_freq 2500→200. Episodes list (488), seed 1000, batch 4,
  π₀-base path, full-fine-tune flags byte-identical to the 047500
  recipe. Also fixed the historical bad `HF_HOME=/c/Users/...` line
  (the makedirs-spy era bug) to `$HOME/Scratch/hf_cache`, and the
  job self-cleans its output dir. Plan on smoke pass: submit 30k
  full train (pi0_v3_out) mirroring v2's actual procedure — 30k
  first, val curve, extend toward 60k only if the curve is still
  improving at 30k (exactly how 047500 was reached).
- **SMOKE ATTEMPT 1 FAILED, ENV NOT MODEL (job 332683, exit 1):**
  lerobot's metadata loader saw the `.cache/huggingface/download/`
  marker that `snapshot_download(local_dir=...)` leaves inside the
  dataset dir (`has_legacy_hub_download_metadata`) and forced a hub
  re-sync, which dies under `HF_HUB_OFFLINE=1`. The relabel's
  `copytree` had carried the marker from the freshly re-downloaded
  v2 mirror (the original writer-created v2 dir was lost in the
  quota janitor sweep — writer-created dirs never have the marker,
  which is why all previous offline trains worked). Fix: deleted
  `.cache/` from both `airvla_v3` and `airvla_v2` local mirrors
  (download bookkeeping only; zero dataset bytes touched).
  Resubmitted as **job 333133** with a fresh log (v3smoke2.log —
  per-job-log rule; first resubmit 333128 reused the old log and
  was qdel'd before start, own job).
- **SOLO MINI PASSED (2026-09-13, job 332011, SOLOMINI-EXIT=0):**
  Plan C on 10 solo pick scenes (seed 99000, PROV `solo: true`):
  **flew-to-target 9/10, target-true median 48.8 mm**, general
  median 77.7 mm, picked 1 placed 1 (ep1: weld tick 295, d_bin
  34.9 mm). Gate (not catastrophic, >4/10 flew-to-target) passed
  decisively — with the distractor physically absent the approach
  band tightens (9/10 within 300 mm vs 47/60 paired). Harness path
  validated ⇒ **4 solo fulls submitted (n=30 picks each, jobs
  333129 047500 / 333130 C-15000 / 333131 ACT / 333132 DP)**, tags
  solofull_47k5/c/act/dp, per-job logs.

### SOLO PROBE FULL RESULTS (2026-09-14, jobs 333129-32, all
### exit 0) — THE CONFOUND DISSOCIATES CLEANLY

Solo scenes (seed 99000, single commanded object, distractor
physically absent), n=30 picks per arm, no assist:

| Arm | Flew-to-target | Gen median | True median | Picked |
|---|---|---|---|---|
| 047500 | 26/30 (87%) | 121.2 mm | 85.8 mm | 0 |
| **Plan C** | **29/30 (97%)** | 98.7 mm | 93.5 mm | 0 |
| ACT | 16/30 (53%) | 275.9 mm | 103.0 mm | 0 |
| DP | 19/30 (63%) | 259.4 mm | 195.5 mm | 0 |

Paired-scene flew-to-target rates for comparison: 047500 70%,
C 78%, ACT 47% (chance), DP 58%.

- **The VLAs' paired-scene shortfall is target selection, not
  approach skill:** remove the distractor and Plan C approaches
  the object 97% of the time (047500: 87%). Its residual paired
  gap (78% vs 97%) is almost entirely wrong-object flights — the
  language-grounding axis, exactly what Plan C/D/D-KI manipulate.
- **The baselines' shortfall is approach competence itself:** ACT
  53% and DP 63% on scenes where there is NOTHING ELSE TO FLY TO.
  Their paired-scene numbers barely move (47→53, 58→63), so
  distractor confusion explains almost none of their deficit —
  answering the examiner question "would the language-blind
  baselines do better with one object?" with a measured no.
- Approach precision is unchanged by the distractor (C true
  median 93.5 solo vs 87.6 paired; 047500 85.8 vs 121.5), while
  the GENERAL median improves sharply solo (C 98.7 vs 145.8) —
  consistent with wrong-target flights inflating the paired
  general median, which is what motivated the target-true metric.
- Picks 0/30 everywhere (mini's 1/10 was the usual rare unassisted
  weld): pure policies still terminal-servo-limited — the solo
  probe isolates approach, not grasp closure; the servo ladder
  covers that axis. (Paired-scene pure pick rates for the record:
  047500 1/60, C 1/60, ACT 1/60, DP 0/60 — scene composition does
  not move grasp closure. A solo run of C+learned servo would give
  the hybrid's grasp ceiling with target selection free — offered
  to Hana, decision open.)

### V3-ARM TRAINING IN FLIGHT (2026-09-14)

- **SMOKE PASSED (attempt 2, job 333133, V3SMOKE2-EXIT=0):** 200
  steps on airvla_v3, loss 0.368 falling, grad norm ~8.8,
  1.2 s/step, checkpoint written and cleaned after. Data path
  (relabeled arm dims through the lerobot loader) validated.
- **FULL TRAIN SUBMITTED (job 333563, v3train.job):** the v2
  baseline recipe verbatim (π₀ base, 30k steps, batch 4, seed
  1000, full fine-tune, same 488-episode list; word-level diff vs
  v2train.job = dataset + output dir only), plus the self-cleaning
  output dir (FileExistsError lesson) and the training_state
  janitor keeping newest two (quota lesson). Started ~01:15,
  ~1.2 s/step, ETA ~11:30.
- **VAL SWEEP CHAINED (job 333599, v3val.job):** submitted only
  AFTER checkpoint 002500 existed (the D incident rule). Rolling
  v2val60k pattern, pinned noise (seed 31415) = exactly paired
  rows, same v2_split.json + v2_manifest_71000.jsonl as the
  047500 selection, but scored against **hanapasta/airvla_v3** so
  dims 3,4 are graded on the real arm deltas (manifest carries
  only metadata; ground-truth actions come from the dataset arg —
  verified before submitting). Output v3_valcurve.json, runs to
  030000.
- **VAL CURVE SO FAR** (overall MSE, real units):
  | ckpt | 2500 | 5000 | 7500 | 10000 | 12500 | 15000 | 17500 |
  |---|---|---|---|---|---|---|---|
  | MSE ×1e-4 | 10.6 | 3.37 | 2.25 | 1.94 | 1.52 | 1.59 | **1.20** |
  Textbook from-base descent; the single 15000 uptick was noise
  (17500 = new best). NOTE: v3 MSE rows are NOT comparable to the
  v2/D/D-KI curves' absolute values — dims 3,4 now carry real
  targets instead of freely-predictable zeros.
- Remaining pre-registered gates: val-curve T4 selection →
  `--policy-arm` eval flag (arm deltas from dims 3,4, clipped
  0.06/tick, replacing q_travel/q_carry switching) + expert-replay
  validation → closed-loop mini → frozen full n=60+20 vs 047500.
- **VAL SWEEP COMPLETE (job 333599, V3VAL-ALL-DONE) — SELECTED
  025000 (6.29e-5), NO EXTENSION:** full curve ×1e-4: 2500 10.6 ·
  5000 3.37 · 7500 2.25 · 10000 1.94 · 12500 1.52 · 15000 1.59 ·
  17500 1.20 · 20000 0.98 · 22500 0.70 · **25000 0.63** · 27500
  0.77 · 30000 0.69. The curve TURNS after 25000 (both later
  checkpoints worse), so unlike v2 (still improving at its 30k
  boundary → extended) the optimum is interior and the mirrored
  procedure resolves: select 025000, no 60k extension. Train
  333563 exit 0 (30k steps, ~11 h, loss healthy throughout).
- **--POLICY-ARM BUILT (2026-09-14):** platform_v2.V2Platform gains
  `policy_arm=` (default False = byte-identical path): dims 3,4
  applied as per-tick arm deltas clip ±0.06, cmd clamped to the
  actuator ctrlrange, REPLACING phase switching; deployed keeps
  updating for metrics. eval_v2 `--policy-arm` flag + PROV field;
  composition with --learned-servo asserted out (not
  pre-registered). Surgical cluster mirror, sha parity all three
  files (platform_v2 090a5ad25ceb, eval_v2 f253176c28c8,
  replay_v3_arm 5f1633ba2ee4), compile-checked.
- **GATED CHAIN SUBMITTED:** replay gate **337932** (CPU,
  replay_v3_arm.py: captures 6 expert episodes, relabels dims 3,4
  in-memory by the relabel_v3 rule, replays through the REAL
  V2Platform policy_arm=True — pass = every expert-grasped episode
  welds+lifts; prints max|dq| arm-tracking error) → mini **337933**
  (10+4, ckpt 025000, tag v3mini, gate V3REPLAY-EXIT=0) → full
  **337934** (60+20, tag v3full, gate V3MINI-EXIT=0). One-variable
  comparison target: 047500's pure row (1/60, 170.8 mm,
  42/60 flew, 9/20 nav).

### E5-DKI SUBMITTED (2026-09-14, job 335659) — D-KI + LEARNED
### SERVO, GATE AMENDMENT

- The composition gate pre-registered for D-KI (flew-to-target
  ≥47/60) was NOT met (43/60). **Hana approved running it anyway
  (2026-09-14, "yes queue it")** on the argument that the learned
  servo's conversion depends mainly on approach precision (capture
  R=0.15 m), which is D-KI's strongest axis (best pure target-true
  88.2 mm, best pure general 117.8 mm), so the composition tests
  precision-vs-grounding trade directly. Recorded as an explicit
  gate amendment, C/D-mini precedent.
- Job v2e5dkifull.job = v2e5dfull.job with ONLY ckpt path
  (pi0_dki_out/checkpoints/005000), tag e5dkifull, log, job name
  changed (diff-audited). Same frozen n=60+20 seed-97000 scenes,
  torchseed 1000, e5_actor_final.pt, --assist 0.15.
- Prediction (registered before result): fewer engagements than
  E5C (wrong-target flights cost opportunities: 43 vs 47) but
  equal-or-better conversion on engaged episodes; plausible range
  for grasps 28-38/60. E5C's 36/60 (60%) is the bar.

### E5-DKI FULL RESULT (2026-09-14, job 335659, E5DKIFULL-EXIT=0)
### — SECOND-BEST SYSTEM; E5C KEEPS THE CROWN

- **PICK n=60: 33 grasped (55%), 26 placed (43.3%)**, general
  median 14.9 mm, **target-true median 9.3 mm = best of the
  entire campaign** (E5C 10.1); flew-to-target **46/60** (E5C 45);
  NAV **12/20** (ties best). Same frozen scenes, same servo, same
  seeds as every rung.
- **Mechanism:** engagements 39 (E5C 40), conversion 33/39 = 85%
  (E5C 90%); median 90 assist ticks on conversions (E5C 71), max
  302; grasp-no-place 7 (E5C 9); wrong-band 14 (E5C 15).
- **Prediction scorecard (registered pre-result):** grasps 33 ∈
  [28,38] ✓; "fewer engagements" 39<40 marginal ✓; "equal-or-
  better conversion" ✗ (85% < 90% — the honest miss). The
  precision advantage DID materialize (9.3 mm true median, best
  ever) but did not convert to more grasps: D-KI's servo hand-offs
  ran slower (90 vs 71 ticks), costing ~3 conversions to the
  clock.
- **Reading:** composition-level grounding is EQUAL (46 vs 45
  flew-to-target — the pure-policy gap 43-vs-47 washed out under
  assist), so the E5C-vs-E5DKI difference is almost purely the
  conversion axis. Insulation buys approach precision and nav
  robustness but the fully-fine-tuned C remains better matched to
  the servo's engagement dynamics. **Final ladder: E5C 45.0% >
  E5-DKI 43.3% > E5D 35% > H1C 36.7%-scripted.** Both
  freeze-vs-full comparisons now have closed-loop answers.

- **Question:** is the system robust to instruction WORDING it has
  never seen? Every training episode used the single template
  "pick up the {obj} and put it in the wooden box"; this is the
  cleanest OOD axis for the instruction-blindness chapter (CAST /
  vision-shortcut literature) and needs no new scene machinery.
- **Harness amendment (frozen-discipline):** new additive
  `--paraphrase` flag in eval_v2.py — cycles 5 unseen phrasings
  deterministically by episode index (ep % 5): "grab … drop it
  into", "pick … up and place it in the box", "put … into",
  "lift … carry it over to", "take … set it down inside". Flag
  absent = byte-identical frozen behaviour. Prompt recorded per
  episode (EVAL `prompt` key) and in PROV (`paraphrase`).
  Surgical cluster edit, sha-verified parity local=cluster
  **7dda224a348b** (pre-edit cluster sha ea80180879ca matched the
  solo run's PROV = we were in sync). Compile-checked.
- **System under test: E5C (the headline system)** — C-15000 +
  learned servo, R=0.15, same frozen seed-97000 pick scenes,
  torchseed 1000. Jobs: **mini 335680** (10 picks, tag paramini)
  → **full 335681** (60 picks, tag parafull, -hold_jid + hard
  gate on PARAMINI-EXIT=0).
- **Reference points (same scenes, canonical prompt):** E5C
  36/60 grasped, 27/60 placed, 45/60 flew-to-target, 12.8 mm
  general / 10.1 mm true median.
- **Prediction (registered):** partial degradation — π₀'s
  PaliGemma backbone should generalize wording far better than a
  from-scratch encoder, but D/D-KI showed grounding is the
  fragile axis; plausible flew-to-target 35-45/60. A collapse
  toward the wrong-object band would be strong instruction-
  brittleness evidence; parity would show the grounding that
  exists is wording-robust.
- Position-OOD arm deferred until this validates (needs careful
  reachability design; registered intent only).

### PARAPHRASE-OOD RESULT (2026-09-14, mini 335680 + full 335681,
### both exit 0) — ZERO WORDING BRITTLENESS

- Mini gate: 6/10 grasped, 5 placed, 7/10 flew, 9.8 mm true —
  PASSED, full auto-launched via -hold_jid gate.
- **FULL n=60 under five never-seen phrasings: 36 grasped (60%),
  28 placed (46.7%), flew-to-target 46/60, general median
  12.0 mm, target-true 10.2 mm.** Canonical-prompt E5C on the
  same scenes: 36 / 27 / 45 / 12.8 / 10.1. **Complete parity on
  every axis** (place and approach a hair above — noise).
- Prediction scorecard: registered 35-45/60 flew-to-target
  anticipating partial degradation; landed 46 — ABOVE the range.
  The degradation hypothesis is falsified: E5C's grounding is
  fully robust to instruction wording.
- **Reading for the thesis:** the paired-command effect is not
  template memorization — π₀'s language backbone carries the
  command SEMANTICS across verb/preposition/structure variation
  (grab/drop, particle movement, truncated "the box", implicit-
  grasp "put the X into..."). Combined with the solo probe this
  completes the grounding decomposition: failures are about WHICH
  object (selection, 14/60 wrong-band unchanged), never about
  WHAT THE WORDS MEAN (wording-invariant) or HOW TO FLY (solo
  97%). Per-episode prompts recorded in the EVAL lines (PROV
  paraphrase=true).

### V3-ARM REPLAY GATE PASSED (2026-09-14, job 337958, exit 0)

- First submission (337932) died on ENV not logic: EGL cannot
  create a headless context on CPU-only nodes (V2Runner always
  builds camera renderers). Chain rebuilt on gpu=1 (337958 →
  mini 337960 → full 337961), fresh log v3replay2.log per the
  per-job-log rule; queued dependents qdel'd before start (own
  jobs).
- **PASS 6/6:** every expert-grasped episode welds + lifts when
  its relabeled actions drive the arm through the REAL
  policy_arm=True platform; miss 6.1-6.6 mm, weld ticks 195-290,
  **max arm-tracking error |dq| < 0.01 rad throughout** — the
  relabel rule, the delta path and the ±0.06 clip compose
  correctly. Contract validated; mini + full may proceed.
- **V3 MINI PASSED (job 337960, V3MINI-EXIT=0):** 10+4 on ckpt
  025000 with --policy-arm: **flew-to-target 8/10, target-true
  median 83.3 mm**, general 117.7 mm, picked 0, nav 1/4 (n=4 —
  noisy, not gating). No flight breakdown with the policy driving
  the arm — the catastrophe gate (≤4/10 flew or flight loss)
  passes decisively; **full n=60+20 auto-launched (337961)**.

### V3-ARM FULL RESULT (2026-09-15, job 337961, V3FULL-EXIT=0) —
### THE ANSWER IS NO: ARM SUPERVISION HURTS AT THIS DATA SCALE

- **v3full (ckpt 025000, --policy-arm, frozen n=60+20):** PICK
  general median **264.2 mm** (047500: 170.8), **target-true
  139.4 mm** (121.5), **flew-to-target 38/60** (42), picked 0
  (1), placed 0 (1); **NAV 7/20** (9/20). Worse than the 047500
  baseline on EVERY axis of the one-variable comparison.
- **Verdict on Hana's question ("shouldn't the VLA control the
  arm?"): measured no.** The relabel already showed the arm
  signal is ~deterministic (99% of deltas at the 0.03 slew cap);
  training on it bought nothing and cost real performance — the
  two extra supervised dimensions act as a noise tax on the dims
  that matter, and the policy's own arm commands add jitter the
  phase-based platform never produces. The mini's healthy-looking
  numbers (8/10, 83.3 mm) were small-n flattery; the full undoes
  them.
- **Offline/online divergence, third instance:** v3's val curve
  was the best-looking of any run (6.29e-5 — deflated by the two
  now-easy near-deterministic arm dims) while its closed-loop is
  the worst pure π₀ arm of the campaign. Validation MSE cannot
  arbitrate design questions; only closed-loop can.
- **Dissertation reading:** the "Arm as Platform, Not as Policy"
  section is now backed by a pre-registered, one-variable,
  n=60+20 experiment instead of an argument. Chapter-ready:
  design claim → obvious objection → measured answer.

### FT-DAG PRE-REGISTRATION (2026-09-14, Hana: "let's try this" —
### DAgger, the strongest untried pure-VLA lever)

- **Question:** does ON-POLICY corrective data — expert
  completions from states the POLICY actually reaches — fix the
  pure VLA's terminal, where hand-designed offsets (F1) improved
  precision but not the grasp rate? Rationale: this exact
  mechanism took the servo from 62% to 90% conversion; the pure
  VLA has the same disease (demos cover only the expert's 3 mm/
  tick corridor; its own slightly-off states are OOD).
- **Method — roll-in policy, roll-out expert (chunk-consistent
  DAgger):** per unit, Plan C (C-15000) drives the frozen-style
  platform on a FRESH collection scene (seed family 81000; trial
  82000 — both disjoint from 71-79k/36k/88k/97k/99k) with NOTHING
  recorded; takeover fires when the jaws hold within 0.25 m of
  the commanded object for 25 ticks ("parked near", the
  policy's real failure state) or at the 600-tick roll-in cap
  ("timeout", covers wrong-object states — recovery-to-commanded
  data). At the seam the expert's hidden integrators are synced
  to the platform (sp, yaw, goals, grip reopened) and the STOCK
  pick_v2 → place_v2 pipeline completes the episode — recovery
  is demonstrated in the standard motion vocabulary, recorded
  frames start AT the seam (policy actions are never supervision;
  roll-in contact counters zeroed at takeover, roll-in hits kept
  in the manifest). Policy-welded roll-ins are discarded.
  Acceptance = the standard _judge_pick gates + banking filter,
  6 attempts/slot, time-budget stop, per-seed manifest
  (dag_manifest_<seed>.jsonl records trigger, seam distance,
  roll-in ticks).
- **Data/training plan:** target ~150 banked units →
  hanapasta/airvla_dag; merge with v21 → **airvla_v24** (split:
  existing retained, new episodes 80:20 as units, seed 424246);
  **FT-DAG = the FT-C recipe verbatim from C-15000** (all dagger
  train episodes + stratified half of the existing mix, 20k
  steps, 5e-6→5e-7, seed 1000) so FT-DAG vs C is one variable:
  on-policy vs hand-designed corrective data.
- **Gates (mini-first):** collection trial (10 units, seed 82000,
  seam/parking/clean audit + film check) → full collection →
  merge read-back → training smoke 200 steps → full train → val
  curve (T4) → closed-loop mini 10+4 → frozen full n=60+20 vs C.
- **Predictions (registered):** target-true median < 87.6 mm
  (beats C); pure grasp rate the primary uncertainty — ≥4/60
  would clearly exit the 0-2/60 band all pure arms occupy;
  null result (still ≤2/60 with better median) would locate the
  residual gap in perception/actuation rather than data coverage,
  making the terminal-observability audit the next lever.

### FT-DAG TRIAL RESULT (2026-09-15, job 338825) — MECHANISM
### VALIDATED; WRITER LOST TO THE WALL (MARGIN LESSON, AGAIN)

- **Every attempted unit banked first-try, 4/4, all clean,
  parking window 1 throughout the seam:** slot 0 timeout-takeover
  from 666 mm (wrong-object recovery flavour) then THREE
  near-trigger seams at **181 / 189 / 213 mm after 174-227
  roll-in ticks** — the expert taking over precisely from the
  policy's characteristic parked-short state, which is the whole
  point of the design. d_bin 13-40 mm on completions.
- **The trial DATASET is unreadable** — the node ran ~1 h/unit
  (lottery), unit 5 overran the 4.5 h budget check into the 6 h
  wall, and the wall-killed LeRobot writer never finalized
  (meta/episodes absent — the documented 2026-09-09 loss mode).
  Cost: 4 trial episodes, nothing else; all gate evidence is in
  dag_manifest_82000.jsonl. Lesson re-learned as arithmetic:
  budget-to-wall margin must exceed the WORST observed unit time
  (1.6 h), not the mean.
- **FULL COLLECTION LAUNCH (amended for node lottery):** four
  parallel collectors, seeds **81000 / 83000 / 84000 / 85000**
  (disjoint from all prior families), n=50 units each, **20 h
  budget inside a 24 h wall** (margin > 2× worst unit), repos
  airvla_dag1..dag4, per-job logs dagcol1-4. Total banked target
  ~100-150 (accept what lands, D-precedent); merge-N follows.
- **COLLECTION PROGRESS (2026-09-15 evening):** three of four
  running (c3/84000 still queued). Manifest ground truth:
  81000 **7 banked**/8 attempts (3 near-seams) · 83000 **3/4**
  (2 near) · 85000 **7/7** (3 near) = **17 banked, 8 near-seams
  (47%)**, zero policy-welded roll-ins, zero writer errors. The
  early 1-of-6 near-seam scare was startup variance; the running
  ratio matches the trial's. Rate ~1.2 h/unit on the slow nodes;
  projection 45-70 total if c3 never starts, ~60-90 if it does.

### WRONG-OBJECT ASYMMETRY ANALYSIS (2026-09-15, derived from
### existing frozen logs — zero new GPU-hours)

Among wrong-object episodes (miss ≥ 300 mm), commanded-object
split (cmd=weight→went-penguin : cmd=penguin→went-weight):
baseline **12:6** · FT-C 7:6 · FT-D 9:11 · FT-D-KI 9:8 · E5C
10:5 · ACT **20:12** · DP 9:16 · solo arms ~0. Reading (pooled
honestly): the baseline's 2:1 penguin-ward bias is **reduced
toward symmetric by paired-command training** (C-policy pooled
across its two runs: 17:11 — reduced, NOT proven eliminated;
small n) · **ACT reproduces the penguin-ward pull with no
language channel** (visual salience: the large high-contrast
plush) · wording-invariant (paraphrase) · absent without a
distractor (solo). Mechanism: a visual-salience prior that
grounding counteracts. Caveat: DP skews the OTHER way (9:16), so
salience is not architecture-universal. Criterion note: this
band definition (≥300 mm) differs from the thesis draft's
terminal-zone definition (16 wrong / 39 entered) — one
definition per table.

### FULL OOD LINEUP PRE-REGISTRATION (2026-09-15, Hana: "do a
### full OOD experimental line up" — robustness of E5C)

- Three arms on **E5C** (C-15000 + learned servo, R=0.15,
  torchseed 1000), each isolating one axis; mini(10) → gated
  full(60); additive flags, PROV-recorded; cluster parity
  verified (eval_v2 612a19be1345, collect_v2 ec7324de6312
  normalized — local CRLF, content identical). Completed
  paraphrase arm = the fourth row of the block.
- **OOD-N `--synonyms` (jobs 345158→345159):** canonical template,
  object NOUNS swapped to never-trained names (toy penguin /
  stuffed penguin / blue plush bird · dumbbell / metal weight /
  calibration weight, ep%3), frozen 97000 scenes = exactly paired.
  Prediction: parity-to-mild degradation (flew-to-target
  40-46/60); collapse ⇒ grounding is trained-token-bound.
- **OOD-P `--oodpos` (345160→345161):** TARGET lateral position
  forced outside the trained band (|x| ∈ [0.36, 0.42] vs ±0.35;
  ≥3 cm table-edge margin; distractor normal; y band unchanged),
  own seed family **96000**. DOCUMENTED: position is informative
  under this flag (target = outer object) — irrelevant to a
  frozen policy; scenes never reusable for training/probes.
  Prediction: modest degradation (grasp 24-34/60) — the servo is
  position-agnostic; risk is the POLICY's approach at unseen
  lateral eccentricity.
- **MERGE ATTEMPT 1 FAILED, ENV NOT DATA (job 347002,
  DAGMERGE-EXIT=1):** all four sources discovered clean (200
  banked), but the local writer-created airvla_v21 had been lost
  in the quota-janitor sweep, so aggregate resolved it through
  the hub cache whose lazy snapshot held only metadata —
  av.FileNotFoundError on the first video. Fix: the merge script
  now ensures a FULL local v21 snapshot first (+ strips the
  legacy download marker, the v3-smoke lesson). Chain rebuilt
  **348338 (merge, log v2dagmerge2.log) → 348342 (smoke) →
  348343 (train) → 348344 (val)**; one sequencing slip (smoke
  submitted before its gate was retargeted) caught and fixed by
  qdel+resubmit before anything ran.
- **Attempts 2-3 (348338, 348363):** each fixed the previous
  failure and found the next: attempt 2 = FileExistsError on
  attempt 1's partial destination (fix: clear the wholly-derived
  destination first); attempt 3 = **aggregation + split SUCCEEDED
  (1,110 episodes / 512,577 frames; train 888 / val 222, dagger
  160/40)** then 401 Unauthorized at the hub push — the job's
  HF_HOME redirect hides the login token; the D-era merges solved
  this with HF_TOKEN_PATH, which v2dagmerge.job lacked. Fix:
  export HF_TOKEN_PATH (pointer only; token never read/printed,
  per the standing rule). **Attempt 4 = chain 348385→348388**,
  idempotent re-aggregate then authenticated push.
- **MERGE COMPLETE (attempt 4, job 348385, DAGMERGE-EXIT=0):**
  **airvla_v24 = 1,110 episodes / 512,577 frames**, split 888
  train / 222 val (dagger 160/40, seed 424246), all five repos
  pushed + read-back verified (v24 revision bd693b8065ee; dag1-4
  banked to the hub too — the collection is now
  outage-safe). Smoke running; train + val chained.
- **OOD-D `--novel-distractor` (345162→345163):** the v1-era
  mustard bottle (in the XML, absent from every v2 training
  frame) dropped on a third table spot by a DETERMINISTIC
  candidate-grid rule (no rng draws ⇒ the frozen 97000 scene
  stream is untouched, exactly paired with canonical E5C).
  Bottle is in no contact-counter set; weld ignores it.
  Prediction: the sharpest risk of the three — a salience-prone
  policy may approach novel clutter; flew-to-target 38-45/60;
  wrong-band composition (bottle-adjacent parks) will be read
  from per-episode positions.
- Queued behind the 4 DAgger collectors; ~8 GPU-h total.
- **OOD-P MINI PASSED (2026-09-15 late, job 345160,
  OODPMINI-EXIT=0) — E5C grasps at unseen positions:** n=10 on
  out-of-band targets (|x| ∈ [0.36, 0.42]): **picked 5, placed 2**,
  flew-to-target 6/10, general median 22.9 mm, true median
  14.7 mm. Even where the target's lateral position exceeds every
  training draw, the system converts half its attempts at mini
  scale. Full n=60 auto-released (345161).
- **OOD-P FULL RESULT (2026-09-16 night, job 345161,
  OODPFULL-EXIT=0) — SPATIAL OOD DEGRADES ACQUISITION, NOT
  TERMINAL SKILL:** n=60 out-of-band targets: **18 grasped (30%),
  12 placed (20%), flew-to-target 31/60 (52%)**, general median
  295.2 mm, **target-true median 11.2 mm** (canonical E5C:
  36/27/45, 12.8/10.1). Reading: once the drone reaches the
  eccentric target the servo converts at normal precision
  (11.2 mm!); the loss is entirely in FLYING TO a target parked
  outside every trained lateral draw (75%→52%). Prediction
  scorecard: grasps 18 < the registered 24-34 range — missed low;
  the acquisition drop was underestimated. Confound note (as
  pre-registered): under --oodpos the target is always the outer
  object, so the salience/selection prior compounds the spatial
  novelty; the wrong-band growth mirrors the flew-to-target drop.

### OOD-N + OOD-D FULL RESULTS (2026-09-16, jobs 345159/345163,
### both exit 0) — THE OOD BLOCK COMPLETES: E5C SWEEPS THREE OF
### FOUR AXES

| OOD axis (n=60 each) | Grasped | Placed | Flew | Gen/true med |
|---|---|---|---|---|
| Canonical reference | 36 | 27 | 45 | 12.8 / 10.1 |
| Paraphrase (wording) | 36 | 28 | 46 | 12.0 / 10.2 |
| **OOD-N (unseen names)** | **37** | **28** | 46 | 11.4 / 9.9 |
| **OOD-D (novel clutter)** | 33 | 23 | 45 | 15.2 / 11.6 |
| OOD-P (unseen positions) | 18 | 12 | 31 | 295.2 / 11.2 |

- **OOD-N: complete parity, nominally BETTER** (37/28) — the
  grounding reads never-trained nouns ("dumbbell", "toy penguin",
  "blue plush bird", "calibration weight") at full performance:
  noun-level generalization, not token binding.
- **OOD-D: essential parity** (33/23/45; prediction 38-45 flew →
  landed 45, top of range) — a never-seen high-salience object on
  the table does NOT hijack target selection, tempering the
  salience-prior concern: the prior shows in wrong-object
  DIRECTION between trained objects, not as capture by novel
  clutter.
- **The OOD story, complete:** robust to instruction wording,
  object names, and novel clutter; the one measured weakness is
  SPATIAL extrapolation (OOD-P: acquisition halves, terminal
  precision intact at 11.2 mm). Language generalizes; geometry
  is bounded by the training envelope.

### DAGGER COLLECTION LANDED THE TARGET (2026-09-15/16, through
### the Myriad login outage)

- **149 banked units, 105 near-seams (70%)** at last count:
  c1/81000 **50/50 COMPLETE in 56 attempts** (COLLECT-DAG-DONE),
  c2/83000 48, c4/85000 50, c3/84000 finally scheduled (1 so
  far, still running). The original 150-unit design target is
  effectively met, with a far richer near-seam fraction than the
  trial suggested. Compute jobs ran through the login outage
  untouched (outage #3, several hours, login-only).
- Next (as collectors exit): 4-source merge → airvla_v24 (split
  seed 424246, read-back gate) → FT-DAG smoke → full train →
  val curve → minis → frozen full vs C.
- **FT-DAG CHAIN SUBMITTED (2026-09-16 night, jobs 347002-347005),
  fully gated:** merge (v2_merge_dag.py — clean-exit-gated source
  discovery per the wall-kill lesson, v21 split verbatim + dagger
  80:20 seed 424246, push + read-back) → smoke 200 steps →
  full train (dag_train_wrapper.py: C-list rebuilt seed 424244 +
  all dagger train episodes, C's recipe verbatim from C-15000,
  self-cleaning output + quota janitor) → val curve (concurrent,
  touches nothing until the first real checkpoint exists —
  FileExistsError lesson — baseline C-15000 as 000000, scored on
  airvla_v24). Merge holds on the three remaining collectors.

### COMP CHAIN + TRUE-PAG DEPLOYED (2026-09-16 after login
### recovery; jobs 346997-347001)

- Files streamed with git-audited deltas (commit 7c552b2) +
  normalized sha parity all three (eval_v2 a1b1abdff108,
  collect_v2 fecc68d91923, comp_setcheck 80fd81759bda); compiled
  on the pinned env.
### C1 COMPOSITIONAL RESULTS (2026-09-16, minis complete — the
### PRE-REGISTERED NULL STANDS, WITH A RICH STAGE DECOMPOSITION)

- **Pure C-15000 (compmini, 347073, exit 0):** crossed **10/10**,
  approached 6/10, picked 0, placed 0. The never-trained composite
  sentence CHAINS both trained behaviours — crossing even beats
  the plain-nav rate (10/10 vs 12/20) — and the missing grasp is
  the standard pure-policy terminal limit, not a composition
  failure.
- **E5C (compminis, 347074, exit 0):** crossed **10/10**,
  approached 5/10, **picked 4/10** (servo converted 4 of 5
  engagements on composite scenes), placed 0 → **composite
  SUCCESS 0/10**. Film-grade decomposition of the 4 grasps: 3
  MID-CARRY DROPS (weld ticks 358/473/616, ended un-welded,
  d_bin 0.7-1.6 m — the documented post-grasp weakness, possibly
  aggravated under the composite prompt: plain-pick carry
  success is 27/36, here 0/3-with-time) + 1 clock truncation
  (weld at 1636/1700). Budget amendment NOT warranted — the
  failures are drops, not clock.
- **Verdict per pre-registration:** zero composite successes in
  either mini ⇒ compfull's null gate exits without the n=60 run.
  **Compositionality moves from "untested" to a STAGED result:
  navigation and approach compose zero-shot; grasp composes with
  terminal assistance; the carry/place phase is where the
  composite breaks** — a far stronger chapter than a bare null.

- **SETCHECK ATTEMPT 1 FAILED 2/6 (job 346997, 2026-09-16
  night) — expert choreography, not harness:** crossing and hover
  6/6 and the staged scoring worked, but nav ends hovering OVER
  the object and pick_v2's standard branch descends to grasp
  altitude AT THAT XY before retreating to its standoff — over
  the table, dragging the legs (87-573 table-hit ticks/episode;
  weight grasps 0/4, forgiving penguin 2/2, one placed-outside).
  Fix: a climb-and-retreat BRIDGE LEG in comp_setcheck.py only
  (room-side standoff at alt+0.15 before the stock pick approach);
  harness/scoring untouched, sha 7e1d5779e0ce mirrored. Pending
  minis qdel'd BEFORE their gates burned (no logs written); chain
  resubmitted as **347072-347075** with fresh log names
  (compcheck2.log).
- Chain: **compcheck 346997→347072** (expert-chain 6/6 gate) →
  **compmini 346998** (C-15000 pure, 10) + **compminis 346999**
  (E5C, 10) → **compfull 347000** (E5C n=60, runs ONLY on any
  composite success in either mini; otherwise exits 0 recording
  the pre-registered null). **pagfull 347001** (E5C + --pag,
  n=60, frozen 97000 scenes, paired vs e5cfull) runs
  unconditionally.

### C1 COMPOSITIONAL PROBE — BUILT (2026-09-15, Hana: "deffo lets
### run this asap"; deployment pending the Myriad login outage)

- Implements the 2026-09-10 pre-registration exactly: `--comp` in
  eval_v2 — nav spawn behind the gate + PICK-BAND objects
  (reset_scene_v2 comp=True), composite instruction = the
  concatenation of the two trained prompts ("fly through the gate
  and hover over the {obj}, then pick up the {obj} and put it in
  the wooden box" — never in any training episode), staged scoring
  crossed/approached/picked/placed, success = crossed AND picked
  AND placed, 1700-tick budget, COMP SUMMARY line, traj banked
  kind="comp". Composes only with --assist/--learned-servo
  (asserted). Scene family **95000** (fresh).
- Gate chain (to submit when the login node returns):
  comp_setcheck.py (6 chained-expert episodes: nav_v2 → pick_v2 →
  place_v2 under the comp scoring, must pass 6/6) → compmini
  (C-15000 pure, 10) → compminis (E5C, 10) → compfull (60) ONLY if
  either mini shows any composite success — zero-shot ~0 is a
  reportable null per the pre-registration.

### TRUE-PAG ARM — BUILT (2026-09-15, the 2026-09-08 redesigned
### experiment; triggered by Hana's "would it be difficult")

- `--pag` in eval_v2: at the weld instant the PD's **total_mass**
  is increased by the payload's subtree mass (and restored at
  release) — the variable the PD's thrust AND arm gravity-moment
  feed-forwards actually read; this is the genuine version of what
  the falsified trim only pretended to do. Guarded against
  --naive-payload; PROV pag=true; weld wrapper mirrors the _ff_on
  latch so double-weld/cleanup paths stay correct.
- Known approximation (documented): the payload's contribution to
  composite inertia and to the CoM offset in the moment FF is not
  modeled — the dominant payload effect (hover thrust) is.
- Plan: **pagfull = E5C + --pag, n=60 picks, frozen 97000 scenes,
  paired against e5cfull**; primary metrics carry sag / settle /
  place accuracy from the banked trajectories, plus the standard
  ladder metrics. Answers Q6 ("does payload compensation reduce
  payload-induced disturbance?") in its honest post-falsification
  form.
- **PAG FULL RESULT (2026-09-16, job 347001, PAGFULL-EXIT=0) —
  GENUINE PAYLOAD COMPENSATION IS BEHAVIORALLY NEUTRAL AT THIS
  PAYLOAD SCALE:** ladder metrics at parity with canonical E5C
  (35 grasped / 28 placed / 46 flew, 12.8 general / 10.4 true vs
  36 / 27 / 45, 12.8 / 10.1). Trajectory mining (36 vs 35 welds):
  post-weld carry sag **median ~0 mm in BOTH arms** (3-tick
  sampling), p90 36.5 vs 43.6 mm, max 138 vs 181 mm; place d_bin
  medians 32.8 vs 68.7 mm — all differences within unpaired
  flow-noise variance. **Reading: at 4-9% mass fraction (45-100 g
  on 1.097 kg) the PD's conditional z-integral absorbs the weld
  transient before feed-forward could matter; the reference
  paper's guidance premise targets the ≥200 g sag regime.** Q6
  answered: no measurable payload-induced disturbance remains for
  compensation to remove at this scale. Caveats: runs are
  policy-stochastic (fresh flow noise, not tick-paired); sag from
  actual altitude at 3-tick stride.

### CARRY-TRAJECTORY METRICS (2026-09-16, mined from banked
### trajectories — no re-run; definitions matter)

- From eval_v2_traj.jsonl (drone+jaw xyz every 3 ticks) + EVAL
  weld ticks, for all four hybrid systems' carried episodes:
  **release proximity** (min jaw-to-box-centre after weld, mm,
  med/p90): H1C 9/66 · E5C 12/570 · E5D 17/891 · E5-DKI 21/385
  (tails = the known mid-carry-drop episodes). **Weld-transient
  dip** (altitude lost ≤120 ticks post-weld — the trim-relevant
  quantity): median ~0 all systems, p90 36.5 (E5C) / 43.6
  (E5-DKI) mm.
- **Definitional finding:** "sag vs the COMMANDED hover altitude"
  is NOT computable from these logs (actual altitude only, no
  setpoint); behavioural proxies conflate loaded-turn settling
  and the deliberate place descent (attained-cruise reference
  yields 124-155 mm medians — not comparable to the expert's
  commanded-reference 5 mm). Thesis options recorded: swap the
  sag column to the weld-transient dip (computed, comparable), or
  a one-flag setpoint-logging re-run for the strict definition.

### DISSERTATION-SUPPORT ERRATA LOG (2026-09-15, figure/bullet
### audit against this ledger)

- Recurring wrong-row slip found TWICE in draft materials: **88.2
  mm used where 87.6 mm is meant.** 87.6 = FT-C pure target-true
  median (the F1/failure-driven-data result); 88.2 = FT-D-KI pure
  target-true median. Adjacent rows in the target-true table —
  easy to grab wrong; both draft figure panel B and a key-findings
  bullet had it.
- Draft figure labels "V3" for what is actually **FT-C** (three
  bars). V3 now names the arm-supervision policy, whose real
  numbers (0/60 picks, 264.2 mm general) contradict the labeled
  bars — must be renamed before submission.
- Draft figure panel B mixed median conventions (pure bars
  target-true, servo bars general). Canonical sets, either usable
  if consistent: general 170.8/145.8/18.2/19.2/12.8 · target-true
  121.5/87.6/12.7/15.1/10.1 (with flew-to-target 42/47/41/46/45).
- Panel-B axis wording: the metric is the episode's CLOSEST
  jaw-to-target distance ("terminal precision"), not pre-handoff
  approach quality — servo bars drop because the terminal leg
  completes, not because the approach changed (trigger radius is
  0.15 m).
- "Quadrupling paired data" → as-built is **3.8×** (80→304); and
  the paired-data val-error claim must be scoped to the fine-tune
  line (D 6.80e-5 vs C-baseline 7.73e-5 on the shared 272-ep set).

### E5C FULL RESULT (2026-09-08, job 305965, exit 0) — NEW BEST
### SYSTEM: THE LEARNED SERVO BEATS ITS SCRIPTED TEACHER

- e5cfull (**C-15000 + LEARNED terminal servo, R=0.15, n=60+20**):
  PICK median **12.8 mm, 36 picked, 27 PLACED — 45.0% complete
  task, 60.0% grasp rate** (Hana's original 60% target, now at
  full n=60 scale); NAV 11/20. Same paired scenes as every rung.
- **Mechanism of the win — engagement conversion 36/40 (90%) vs
  the scripted servo's 24/39 (62%)**: DAgger training on
  off-trajectory states (the tube, the come-back labels) cured the
  heading brittleness that stalls the frozen-axis script. The
  student beats the teacher on exactly the axis the teacher's
  latched design cannot handle.
- Costs, honestly: the learned servo is slower (median 71 assist
  ticks on conversions, max 302, vs the script's ~50), yielding
  **9 grasp-no-place losses** (vs scripted 2), FILM-VERIFIED
  (contact sheets in Reports/v2_eval_videos/e5cfull/) as two
  categories: (1) clock losses at the finish line — eps 4/49
  carried the object TO THE BIN MOUTH (d_bin 156-159 mm) and ran
  out of episode after late welds (~tick 886); (2) mid-carry
  DROPS — eps 23/29/33/50 (d_bin ~1.1-1.25 m) grasped cleanly
  (ep29 weld tick 221) then lost the object en route: the weld
  releases only when the grip command re-opens past 0.8, and
  post-weld control is handed back to the POLICY, whose post-grasp
  grip behaviour is essentially untrained (pure C almost never
  welded). Candidate fix reserved (H1.2/E5.1): latch grip closed
  during carry until bin proximity. Wrong-target band 15/60 (same
  policy, consistent with C's 13-14).
- **THE FROZEN n=60 LADDER, FINAL FORM:** pure 1.7% → +scripted
  servo 31.7% → C+scripted 36.7% → **C+learned servo 45.0% (60%
  grasp)**. Both terminal-controller designs (H1 scripted, E5
  learned) are complete rungs; the learned one is the headline.

### PAG FALSIFICATION (2026-09-08): THE PAYLOAD FEED-FORWARD IS
### INERT IN THE PD-ERA STACK

- Building the pre-registered `--naive-payload` ablation exposed
  that the mechanism it would ablate does nothing: ground-truth
  replay with the trim verifiably engaged vs bypassed produced
  MILLIMETRE-IDENTICAL trajectories (weld ticks 292/195, carry
  z-errors equal to the mm). Code path confirms: V2Runner.weld_grasp
  bumps `nominal_hover_thrust` (collect_v2 L131) but pd_flight
  computes thrust from `self.total_mass` (L170) and never reads
  that attribute — only the retired MPPI warm-start paths do. The
  mechanism was live in the MPPI era and was silently orphaned by
  the PD migration.
- Consequences: (a) NO banked number changes — every demo and eval
  flew with the trim inert, identically; (b) payload transfer in
  the shipped system is actually handled by the PD's conditional
  z-integral + the expert's measured lead compensation (the
  "loaded PD parks ~0.28 m past setpoint" law); (c) the
  dissertation passage "feed-forward in place of a staged lift"
  MUST BE REWORDED — it credits an inert mechanism (flagged to
  Hana); (d) the pre-registered naive-payload ablation is moot —
  the shipped system IS the naive condition. Redesigned experiment
  (offered, awaiting Hana): a `--pag` arm that updates
  `total_mass` at the weld instant (true payload-aware control)
  vs the shipped stack, measuring carry sag/settle/table-drag/
  place accuracy. The inert `--naive-payload` flag stays in
  eval_v2 with PROV recording, harmless and documented.

### PLAN D COLLECTION AMENDMENT (2026-09-08): collection measured
### at 5.4 min/ep (2× the E3-era estimate; ETA 52 h > 30 h wall),
### so the 300 pairs are split across two parallel jobs: 305829
### (airvla_v2_d, seed 75000, runs to wall-truncation ≈165 pairs)
### + 306679 (airvla_v2_d2, seed 76000, 135 pairs). v2_merge_d
### generalized to both repos; pair units keyed by
### (collector_seed, pair_id).

### PLAN D DATA-LOSS INCIDENT + RECOVERY (2026-09-09):
### WALL-KILLED LEROBOT WRITERS LOSE THE WHOLE DATASET

- Job 305829 (seed 75000) hit its 30 h wall at 456 banked episodes
  (228 pairs). The merge then failed loudly ("Parquet magic bytes
  not found in footer"): the LeRobot v3 writer concatenates the
  ENTIRE dataset into single parquet files (data/chunk-000/
  file-000.parquet + the episode index) whose footer is written
  only at close — a hard kill leaves every episode unreadable, not
  just a torn tail. All 455 committed episodes of airvla_v2_d are
  unrecoverable by standard tools (footer reconstruction = page-
  level thrift parsing, not attempted); repo abandoned. Job B
  (airvla_v2_d2, 270 eps / 135 pairs) closed cleanly and is intact.
- **BANKED LESSON: never point a LeRobot collection at a target
  that exceeds the wall — the writer must reach close(). Clean
  DCOLLECT-EXIT=0 is now a hard merge precondition for every
  source.**
- Recovery: recollection job 315675 (airvla_v2_d3, seed 77000,
  120 pairs = 240 eps, sized ~17 h ≪ 30 h wall). Rebuilt chain:
  merge 315678 (v21 + d2 + d3 → airvla_v22, both-clean-exit gate)
  → train 315679 (unchanged recipe). Final Plan D dose: 255 pairs
  (with E3's 80: 335 total; pre-registration's 300 D-pairs amended
  to 255, documented here). Cost: ~1 day of calendar.

### FINAL-PHASE ROADMAP PRE-REGISTRATION (2026-09-10, Hana's
### directive: "then lets run compositional, then ACT (and
### Diffusion Policy), ignore OOD for now"; both clusters in
### parallel, mini-first, per-task commits)

- **Ordering:** Plan D results → C1 compositional probe → ACT
  baseline → Diffusion Policy baseline. OOD (novel objects/gates)
  DROPPED — stays a documented exclusion.
- **C1 — compositional zero-shot probe (eval-only, pre-registered
  here):** the composite task ("fly through the gate, then pick up
  the {obj} and put it in the wooden box") is DELIBERATELY absent
  from all training data. New eval episode kind in eval_v2: start
  from the nav spawn behind the gate, composite instruction,
  success = clean gate crossing AND pick AND place; stage metrics
  reported separately (crossed / approached / picked / placed).
  Harness validated FIRST by chained expert replay (expert nav
  phase + pick phase on the same scene must complete the composite
  under the new scoring). Mini n=10 on C-15000 (best flier), then
  n=10 with +learned servo; full n=60 only if the mini shows any
  composite success (zero-shot may legitimately be ~0 — a null is
  a reportable result for a held-out composition).
- **ACT baseline (from-scratch imitation, pre-registered):**
  lerobot 0.6 ACT on airvla_v21, SAME train split as Plan C
  (e3_split train, 728 eps), chunk_size=50 / n_action_steps=50
  (naive-50 cadence parity), default ACT optimizer, 50k steps,
  save every 5k, seed 1000. Trains on SPARKS (GB10) in parallel
  with Myriad's Plan D. Selection: lowest val-split MSE (adapted
  curve) with final-checkpoint fallback; then the standard gated
  mini → full n=60+20 on the frozen protocol (pure; +servo arm
  optional afterward). Smoke-first: 200-step mini-train + loss
  sanity before the real run.
  **Stated before any ACT result exists:** vanilla ACT is NOT
  language-conditioned — it ignores the instruction. The baseline
  therefore measures the ceiling of pure visuomotor imitation on
  this task; ~chance target selection (~50% wrong-object) is
  EXPECTED by construction, and the comparison against π₀'s
  wrong-object rate is the designed argument for why a
  language-conditioned VLA is needed at all.
- **Diffusion Policy baseline:** same dataset/split/cadence
  parity, lerobot diffusion policy defaults, same gates; runs
  after ACT on whichever cluster is free.
- **ACT MINI RESULT (2026-09-11, job 321042, tag actmini, exit
  0):** n=10 median **312.6 mm, 0 picked, 0 placed**; NAV 2/4. As
  pre-registered for the language-blind baseline: flies plausibly,
  approaches far worse than any π₀ (baseline 047500: 170.8 mm),
  converts nothing. Training: 50k steps on Sparks GB10 (~5 h),
  final loss 0.080, checkpoint hanapasta/act_v21. Full n=60+20
  dispatched: job 321544, tag actfull. (Ops notes: checkpoint
  transfer via HF; a nested-folder slip crashed the first mini at
  config parse — path fixed; eval amendment `--policy act` with
  native camera keys, PROV records policy_type.)
- **DP TRAINING DONE (2026-09-11):** 50k steps on GB10 (~7.5 h),
  final loss 0.005, checkpoint pushing to hanapasta/dp_v21; mini
  next after ACT full.
- **ACT FULL RESULT (2026-09-11, job 321544, exit 0, frozen
  n=60+20):** PICK median 313.5 mm, **1 picked, 1 placed (1.7%)**;
  NAV 10/20; **TARGET-TRUE: flew-to-target 28/60 (46.7% —
  chance-level target selection at n=60) with 150.3 mm true
  median**. The pre-registered prediction measured exactly: the
  language-blind from-scratch policy navigates comparably (10/20
  vs π₀'s 9-12/20), selects the commanded object at coin-flip
  rate, and approaches worse even when right (150.3 vs π₀'s
  121.5 true median). The single complete success is the coin
  landing right once. The baseline argument for language
  conditioning is complete; π₀'s pretraining buys both grounding
  (42-47/60 vs 28/60) and precision.
- **DP FULL RESULT (2026-09-12, job 324360, exit 0, frozen
  n=60+20):** PICK median 249.8 mm, **0 picked, 0 placed**;
  **NAV 0/20** (never crosses the gate); TARGET-TRUE: flew-to-
  target 35/60 (within noise of chance; binomial p≈0.12), true
  median **115.2 mm**. Profile vs ACT: better approach precision
  near objects (115 vs 150 mm — diffusion's multi-modal fitting),
  but no grasps at all and no navigation. Both from-scratch
  baselines: language-blind (chance selection), grasp-free, and
  each worse than π₀ on at least one axis π₀'s pretraining
  provides. Mini was 0/10, med 365.8, nav 0/4 (obs-history
  harness amendment validated there).
- **Latency (Table 13.1) — MEASURED (2026-09-10, Sparks GB10,
  fp32, full per-inference path incl. tokenization, 100 calls
  after 5 warmup):** median **235.9 ms**, p95 **237.9 ms**, max
  239.2 ms — a very tight distribution. Amortized per 10 Hz
  control tick (one inference serves a 50-tick chunk):
  **4.72 ms**. Reading vs a 150 ms watchdog: a synchronous
  per-inference deadline is NOT met; amortized/pipelined operation
  clears it with ~47× margin (inference fits well inside the 5 s
  a chunk buys, ~5% duty cycle). Weight values do not affect
  latency, so the v1-era checkpoint used is representative of all
  π₀ configurations in the ladder.

### GROUNDING PROBE RESULT (recovered from cluster logs
### 2026-09-12 — previously pre-registered and run, but never
### banked as a result block)

Offline probe (groundprobe.py, jobs 290283 / v2gprobec): on val
PAIR episodes the SAME observation is presented with BOTH commands
under pinned noise; the 50-step integrated xy displacement is
classified toward the correct / wrong object by cosine against the
layout's object directions. FLIP = the fraction of pairs where the
heading actually swaps with the instruction — the direct test of
instruction sensitivity.

| Checkpoint | correct | wrong | null | **FLIP** |
|---|---|---|---|---|
| 000000 (unmodified 047500 baseline) | 0.19 | 0.53 | 0.28 | 0.03 |
| Plan-C 012500 | 0.22 | 0.50 | 0.28 | 0.00 |
| **Plan-C 015000 (selected)** | 0.25 | 0.44 | 0.31 | 0.03 |
| Plan-C 017500 | 0.22 | 0.47 | 0.31 | 0.00 |
| Plan-C 020000 | 0.28 | 0.41 | 0.31 | 0.03 |

- **Reading:** the paired-command data moved the bias in the right
  direction — wrong-object headings 0.53 → 0.41-0.44, correct
  0.19 → 0.25-0.28 — but the **FLIP rate stayed at ~0.03 (i.e.
  essentially zero) at every checkpoint**: swapping the
  instruction over an identical observation almost never swaps
  the policy's early heading.
- **Why this does NOT contradict the closed-loop 47/60**
  (flew-to-target for the same C-15000): the probe integrates
  only the first 50 steps from the episode START, where the two
  objects often lie in a similar direction and ~30% of cases are
  unclassifiable ("null") — it is an early-phase, low-sensitivity
  instrument. Discrimination evidently happens later in the
  approach, which the probe cannot see.
- **Status: SUPERSEDED** by the closed-loop TARGET-TRUE metric
  below (frozen protocol, all runs, measures what the drone
  actually does). Cite flew-to-target for grounding claims; the
  probe is reportable as methodology that proved too insensitive,
  with its one durable finding being the near-zero flip rate.

### TARGET-TRUE METRICS (defined 2026-09-11, Hana's directive:
### separate approach precision from target selection)

- **Definition:** flew-to-target = episodes whose closest approach
  to the COMMANDED object is < 300 mm (below the object-separation
  band — the established wrong-object criterion); target-true
  median = median closest approach over those episodes only. The
  general median stays reported alongside (it mixes both effects:
  a wrong-object flight contributes its large distance to the
  commanded target). Harness amendment: eval_v2 now prints a
  TARGET-TRUE line after every PICK SUMMARY (additive only).
  Script median convention (mm[n//2]) throughout — the general
  column below reproduces every banked number exactly.

| Run | n | General median | Flew-to-target | Target-true median |
|---|---|---|---|---|
| full47k5 (pure 047500) | 60 | 170.8 | 42/60 | **121.5** |
| e3cfull (pure Plan C) | 60 | 145.8 | 47/60 | **87.6** |
| h1full (047500+servo) | 60 | 18.2 | 41/60 | **12.7** |
| h1cfull (C+scripted servo) | 60 | 19.2 | 46/60 | **15.1** |
| e5cfull (C+learned servo) | 60 | 12.8 | 45/60 | **10.1** |
| actmini (ACT baseline) | 10 | 312.6 | 5/10 | 180.4 |
| mini47k5 | 10 | 175.5 | 6/10 | 110.8 |
| e3cmini | 10 | 109.3 | 8/10 | 108.8 |
| h1mini | 10 | 20.8 | 6/10 | 8.6 |
| h1cmini | 10 | 8.9 | 8/10 | 7.3 |
| e5cmini2 | 10 | 15.3 | 8/10 | 10.7 |

  Readings: Plan C's data improved BOTH axes (flew-to-target 42→47,
  target-true 121.5→87.6); the servos' precision is even better
  than the general medians showed (target-true 10-15 mm); and the
  language-blind ACT baseline reaches the commanded object almost
  exactly at chance (5/10) with poor precision even then (180.4) —
  the pre-registered prediction, now measured.

### CONDITIONAL LADDER — success GIVEN correct-target heading
### (right-target = episode ended <300 mm of the commanded object)

| System (frozen n=60) | Right-target | Grasped of those | Placed of those |
|---|---|---|---|
| Pure π₀ 047500 | 42/60 | 1/42 (2.4%) | 1/42 (2.4%) |
| Pure Plan-C | 47/60 | 1/47 (2.1%) | 1/47 (2.1%) |
| 047500 + servo | 41/60 | 21/41 (51.2%) | 19/41 (46.3%) |
| **Plan-C + servo (H1C)** | **46/60** | **24/46 (52.2%)** | **22/46 (47.8%)** |

  Reading: on correct-target flights the hybrid grasps ~52% and
  places nearly all of them; the raw 36.7% is dragged down by the
  wrong-target flights. Plan C's paired data cut wrong-target
  episodes 18-19 → 13-14 (~25% relative), consistent across the
  pure and hybrid runs (same policy flies in each pair) — the
  evidence that scaling paired-command data attacks a movable
  number.

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
