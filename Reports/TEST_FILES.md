# Test-file catalogue

Every script used to run or validate an experiment, why it exists, and how it
measures success. Companion to the decision log (`AIRVLA_RECREATION.md`) and
the results ledger; raw outputs live in `Reports/eval_logs/`.

## How success is measured (shared definitions)

These definitions are what the numbers in every log mean. Where a word is
looser than it sounds, that is stated here rather than discovered later.

| Metric | Definition | Caveat |
|---|---|---|
| `miss_mm` | Closest jaw-to-target approach over the episode, in mm. **The primary metric everywhere.** | Distances carry the evidence; the ~10 mm grasp window makes success counts near-coin-flips when the distribution straddles it. |
| `picked` | **Latched lift**: the grasp weld was active AND the object was >12 cm above the table surface for **at least one tick**. | Includes momentary grab-and-drops. Video review of the frozen n=60 run found 2 sustained carries (~40 s and ~58 s) and 1 transient grab among its 3 `picked` episodes. A carry-duration metric needs per-tick weld/object logging — flagged for the next frozen protocol version. |
| `placed` / `success` (pick) | Object inside the bin at episode end with the weld released. | — |
| `success` (nav) | Clean gate crossing (no gate contact) then hover within 0.25 m over the object sustained 3 s. | — |
| `success` (comp) | Gate cross → hover → pick → place, in that order, no gate hit. | Grasp before gate counts as failure (paper's ordering rule). |
| Rates | Always with 95% CI; no rate under n=40 is treated as a comparison. | Pre-registered in D48. |

## Evaluation harness

| File | Why it exists | What it measures |
|---|---|---|
| `Sim'n'Real/Mujoco/eval_pi0.py` | The working evaluation harness: scene sampling, the repaired platform automaton (`--platfix`), matched start positions (`--startfix`), all intervention flags (`--exech`, `--oracleyaw`, `--oracleyaw2`, `--oraclestate`, `--oraclecorrupt`), seeding (`--torchseed`), video (`--videps`), tagging (`--tag`), PROV provenance line, per-tick trajectory logging. | All episode metrics above, per episode as `EVAL {json}` lines. |
| `Sim'n'Real/Mujoco/eval_frozen_v1.py` | Frozen protocol v1 (sha `a8cc240ba921`). Re-runs must use a frozen copy so numbers stay comparable; the working file keeps evolving. | Same as eval_pi0 at its freeze point. **Lacks `--oracleyaw2`** — flags added after a freeze are silently ignored (the seed-2000 incident). |
| `Sim'n'Real/Mujoco/eval_frozen_v2.py` | Frozen protocol v2 (sha `1ec61ae930ca`), cut to add `--oracleyaw2` after the v1 flag gap was caught via the PROV argv. | Same, plus the weld-releasing oracle-yaw variant. |

## Harness validation (is the test rig itself trustworthy?)

| File | Why it exists | What it measures |
|---|---|---|
| `harness_audit/open_loop_probe.py` | Teacher-forced probe: policy sees training images, no simulator. Separates "policy learned nothing" from "harness blocks everything". | Correlation between predicted and expert actions per channel (~0.9 at 30k = the policy learned the task's structure). |
| `harness_audit/replay_diag2.py` | Ground-truth replay: the expert's own recorded actions fed through the evaluation rig on a byte-faithful world snapshot. If the expert fails, no policy can pass. **The acceptance test for every harness patch.** | Pass/fail of expert replay through the rig; localizes which gate eats the episode. v1 was RNG-unfaithful (reset re-drew ~11 values) — v2 snapshots full state at the first expert tick. |
| `harness_audit/fixfit5.py` | Iterative fitting of the platform repair: demo-calibrated arm poses, contact-debounced weld (2-of-4), tuck timing. | Expert replay success under candidate repairs (3/3 required before the repair became `--platfix`). |

## Failure attribution and analysis

| File | Why it exists | What it measures |
|---|---|---|
| `harness_audit/audit_analysis.py` | Post-audit attribution squares: harness vs start-distribution vs policy contributions. | Miss-distance distributions per condition from `eval_traj` logs. |
| `harness_audit/powered_corr.py` | Lateral-tracking decomposition after the "45% yaw" retraction. | Nose-on-bearing regression, jaw/body slopes (with the kinematic-identity trap documented). |
| `harness_audit/camera_pixels.py` | Observation-deficit measurement: how many pixels the target occupies per camera vs distance. | Segmentation-rendered target pixel counts at 224 px, binned by distance (camera3 dead 0–4 px; camera2 thin at commitment range). |
| `harness_audit/verify_ledger.py` | Recomputes every number on the results ledger from raw logs before publication. | 60 claims: 52 verified as printed, 8 corrected or re-labelled (`eval_logs/VERIFICATION.md`). |

## Interventions and probes

| File | Why it exists | What it measures |
|---|---|---|
| `harness_audit/build_oracle.py` | Builds the oracle dataset (13-dim state = 10 + FK aim−jaws vector) for the oracle-vector experiment, with hardlinked videos and a LeRobot load smoke test. | Dataset integrity (load test); the experiment itself runs through the frozen harness. |
| `Sim'n'Real/Mujoco/forgetting_probe.py` | Language-grounding probe (D50): does the fine-tuned policy still read object nouns? 2×2 of {prompt: bottle/weight} × {normal/swapped positions}, mustard bottle as the pretraining-known probe object. | Closest approach to prompted object; selection rate (which object the jaws end nearer). Success is deliberately not scored — the weld is bound to trained objects. |
| `cam3_rerender/demo_cam3v2.py` | Pre-flight framing demo for the parked camera3 re-render (D49): current vs candidate view, rendered over logged states, before any cluster work. | Visual framing only; on-frame caveats state the two approximated quantities (table u_off, bin tints). |
| `dataset_tests/` (2026-08-31) | Two dataset-only tests deciding whether a v2 collection campaign is justified: expert-reversal analysis and position-shortcut analysis. Pre-registered decision rules written before the numbers. | See `dataset_tests/REPORT.md`. |

## Provenance rules that apply to all of the above

- Every run prints a `PROV` line (script sha, checkpoint, argv, seeds, time)
  before results; it is how the seed-2000 flag incident was caught.
- Log filenames and `--tag` values are never reused; superseded results are
  kept and marked, never overwritten.
- Frozen-protocol runs use the frozen script only; new flags require a new
  freeze (v1 → v2), never an edit in place.
