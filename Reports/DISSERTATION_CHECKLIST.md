# Dissertation experiment checklist — living tracker

Created 2026-09-01. Two parts: (1) the regularization/training-technique
audit (are we doing X?), (2) the full experiment matrix (D/T/E/A/M).
Annotate each item **below its entry** when done, with the result and a
one-line explanation. Status legend: ✅ done · 🔶 partial · ⬜ todo.

**The training code**: there is no custom training script in this repo —
training is the stock LeRobot trainer, `lerobot/scripts/lerobot_train.py`
inside the training environments (`~/skyenv` on Sparks, `env-pin` on
Myriad), invoked as `lerobot-train` and configured entirely by CLI
arguments. The as-run configurations live in the cluster job scripts
(Myriad `~/Scratch/airvla/train.job`, `train_resume.job`,
`train_ext90.job`; Sparks `train_swap_slurm.sh`) and are echoed in each
run's log. Model/optimizer defaults come from
`lerobot/policies/pi0/configuration_pi0.py`.

---

## Part 0 — Regularization & optimization audit (verified in the training env, 2026-09-01)

| Technique | Doing it? | Detail |
|---|---|---|
| Weight decay on the action head | 🔶 global, not head-specific | AdamW `weight_decay=0.01` (inside the suggested 1e-2..1e-4 band) applied to **all** trainable parameters — LeRobot π₀ default; no per-module grouping. |
| Dropout on pre-head projections | ⬜ no | `modeling_pi0.py` contains zero dropout layers; none added. |
| Low LR for the head | ✅ yes (whole model) | `2.5e-5` peak, 1,000-step warmup, cosine decay to `2.5e-6` — inside the suggested 1e-5..1e-4; observed in every training log. |
| LoRA / partial trunk unfreezing | ⬜ no — we do the opposite | **Full fine-tune**: `freeze_vision_encoder=false`, `train_expert_only=false`; the whole trunk adapts. (The pickhold side-project trains the frozen-trunk variant `train_expert_only=true`, so both regimes exist in the project family.) |
| Visual augmentation at train time | ⬜ no | LeRobot's `--dataset.image_transforms` never enabled. We rely on **collection-time** scene randomization instead (positions, yaws, wood tints, table offset) — a different mechanism with different coverage. |
| Proprioceptive/state noise | ⬜ no | Clean states in, clean states trained. |
| Language prompt variations | ⬜ no | One fixed template per task (one held-out phrasing used at eval only). Given D50 (prompt behaviourally inert) + the position test (language was necessary), prompt variety is a live v2-training candidate. |
| Co-training with generalist data | ⬜ no | Fine-tune sees only our episodes. Flagged (Route A discussion, 2026-08-31) as a third training arm after v2: v2-only vs v2-co-trained, same frozen eval. |
| Early stopping on validation | ⬜ no for v1 | No split existed (documented limitation); checkpoints at fixed steps. → items D4/T1/T2. |
| Flow-matching time-horizon noise | ✅ stock π₀ | Flow time τ sampled from a Beta distribution (`sample_beta(α, β)` scaled/offset in `modeling_pi0.py`) — π₀'s own scheme, unmodified. Inference-side sampling is now always seeded (`--torchseed`), after measuring the unseeded variance floor. |

---

## Part 1 — Dataset (D)

**D1 — Expert reliability** 🔶
v1 banked only episodes passing the 18-gate quality harness, so the
dataset contains no failures — but attempt-level statistics (how often
the expert needed retries/rejections) were not systematically kept.
> **Annotation**: v2 collector records per-attempt outcomes including the
> new table-clip gate; expert success-by-task table to be produced from
> the v2 collection log. ⬜ pending v2 collection.

**D2 — Anti-artefact validation** ✅
The strongest existing asset: five evaluation-harness defects proven by
ground-truth replay (expert actions 0/3 through the broken rig, 3/3 after
repair with 3–13 mm placement); collection-side guards (upright-path
weld, object_settled) prevent false-positive grasps.
> **Annotation**: documented in D46, `HARNESS_VALIDATION_DRAFT.md`,
> ledger "harness" tab; before/after table committed. Done 2026-08-29/30.

**D3 — Dataset diversity, quantified** ✅ (v1)
> **Annotation**: `dataset_tests/REPORT.md` — spawn statistics table
> (mean/std/range for target & distractor), scatter plot, overlap
> measures, plus the decisive position-shortcut analysis (52.3% = chance).
> v2 re-run is a pre-training gate.

**D4 — Train/validation/test split** 🔶
v1 trained on all 400 (documented asterisk). 40-episode holdout defined
late; used by the swap arm and v2 plan.
> **Annotation**: v2 target — 480 collected → 400 train / 80 validation
> (by episode), plus an independent seeded test suite through the frozen
> eval. Checkpoint selection by validation loss (→T2). ⬜ pending v2.

**D5 — Leakage check** 🔶
Seed inventory exists (`harness_audit/README.md`); episode-level splits;
eval scenes drawn from a disjoint seed family (77000-series vs
collection seeds).
> **Annotation**: formal leakage checklist (seed reuse scan, near-duplicate
> scene scan, OOD exclusions) to be run and committed with the v2
> dataset. ⬜

## Part 2 — Training (T)

**T1 — Convergence curves** 🔶
Training loss logged every step in run logs (e.g. 90k final loss 0.039);
no validation loss (no split).
> **Annotation**: v2 training adds `--dataset.episodes` split + periodic
> validation-loss evaluation; plot committed per run. ⬜

**T2 — Checkpoint selection rule** 🔶
Current rule: fixed step counts (30k/60k/90k), never selected on test
results — but also not selected on validation (none existed).
> **Annotation**: v2 rule, stated in advance: lowest validation action
> loss. ⬜

**T3 — Per-action-dimension learning** 🔶
Teacher-forced probe measured per-channel prediction correlations (~0.9
overall; dyaw the sparsest channel, non-zero in 13% of frames) — on
training episodes.
> **Annotation**: repeat on the v2 validation split for an honest
> per-dimension table. ⬜ (v1 result: `open_loop_probe.py`, D47.)

**T4 — Dataset scaling (100/200/400)** ⬜
Not done (step-scaling 30/60/90k exists, data-scaling does not).
> **Annotation**: —

**T5 — Corrective-data ablation** ⬜
v2's design makes this cheap: train with and without the 80 corrective
episodes via `--dataset.episodes`.
> **Annotation**: —

## Part 3 — Main evaluation (E)

**E1 — Naive vs RTC** 🔶
Preliminary and cautionary: the horizon intervention (`--exech`, RTC-like
more-frequent replanning) made the v1 policy dramatically *worse*
(412 mm vs 223 mm) — replanning interrupts commitment. Any RTC claim must
reckon with this measured inversion.
> **Annotation**: full three-condition comparison on the v2 policy. ⬜

**E2 — RTC vs RTC+PAG** ⬜ (PAG not yet implemented in this recreation)
> **Annotation**: —

**E3 — 45 g vs 100 g payload (mechanistic)** ⬜
> **Annotation**: —

**E4 — Disturbance metrics before success (sag, recovery, placement)** 🔶
Trajectory logging (per-tick body+jaws) supports these; sag/settling not
yet computed. The v2 eval adds per-tick weld state + object z (from the
picked-flag review) which enables carry metrics.
> **Annotation**: —

**E5 — Control flow-matching stochasticity** ✅ method in place
Unseeded π₀ sampling measured: paired clean-vs-clean action distance
0.600 ± 0.059 (the noise floor that invalidated an unpaired ablation);
`--torchseed` seeds every eval; `sample_actions(noise=)` supports pinned
noise (bit-exact repeat measured 0.000000).
> **Annotation**: use pinned noise for paired RTC/PAG trials. Method
> validated 2026-08-31/09-01 (`PROVENANCE.md`, pickhold side).

**E6 — Confidence intervals** ✅ discipline adopted
> **Annotation**: every rate reported with 95% CI; no rate under n=40
> treated as a comparison; distance distributions primary. In the ledger
> methods note and D48.

**E7 — Paired statistics (McNemar)** 🔶
Paired-by-seed design in place (Runner seed pairs scenes across arms);
formal McNemar not yet run.
> **Annotation**: —

**E8 — OOD object** 🔶
The mustard-bottle probe (D50) is an OOD-object test: flight remained
in-family (takeoff 20/20) but selection ignored the noun — the OOD
result is entangled with the language failure.
> **Annotation**: repeat on the v2 policy, which trains with
> position-uninformative language-necessary data. ⬜

**E9 — OOD spatial** 🔶
The v1 start-distribution finding (0/225 overlap) is a cautionary OOD
result: the policy was unintentionally evaluated OOD and failed badly.
Deliberate, graded spatial OOD eval not yet run.
> **Annotation**: —

**E10 — Held-out compositional task** 🔶
The comp task (gate → pick → place, with the ordering rule) exists in the
harness and is excluded from training; only evaluated pre-audit (invalid).
> **Annotation**: honest comp evaluation on the v2 policy. ⬜

**E11 — Failure taxonomy** 🔶
Deep attribution exists for the pick failure (harness → start
distribution → heading channel → terminal parking → selection lean), but
not as a per-episode labelled taxonomy chart.
> **Annotation**: add per-episode failure-stage labels to the v2 eval
> output; stacked bar per condition. ⬜

**E12 — Latency / watchdog** ⬜
> **Annotation**: —

**E13 — Trajectory quality metrics** 🔶
Miss-distance distributions everywhere; scene-disturbance measured
(object moved >20 mm in 46/60 episodes); path length/sag/settling not
yet standard outputs.
> **Annotation**: —

## Part 4 — Analysis (A)

**A1 — No test-set tuning of guidance** ⬜ (with the pre-registration
machinery already in place, this is procedural: tune on validation,
freeze, then test.)
> **Annotation**: —

**A2 — Guidance-strength ablation** ⬜
**A3 — Guidance-schedule ablation** ⬜
**A4 — VJP numerical validation** ⬜
> **Annotation**: —

**A5 — Causal PAG on/off test** ⬜ (pinned-noise pairing from E5 makes
this clean when PAG exists.)
> **Annotation**: —

**A6 — Negative control (45 g)** ⬜
> **Annotation**: —

## Part 5 — The big ML experiment (M)

**M1 — Pretrained vs scratch** ⬜
> **Annotation**: candidate third training arm alongside the co-training
> arm; same data, steps, optimizer; frozen eval. —

---

## Standing methodology already satisfied across the matrix

Pre-registration of decision rules before results (D48 onward) ·
provenance lines on every run · frozen evaluation protocols (v1/v2, v3
pending for `--swapcams`) · never-reuse rule for tags/logs · corrections
filed at source with an index (`LESSONS.md` Part 3) · paired scenes
across seeds · videos retained for every scored claim.
