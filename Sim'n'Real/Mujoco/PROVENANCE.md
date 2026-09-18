# Provenance — analysis run `run_20260831_ckpt060000`

Checkpoint under analysis: `pickhold_pi0_frozen_s1/checkpoints/060000`
(frozen-trunk π0, `train_expert_only=true`, 60 000 steps, batch 4, seed 1000,
`lerobot/pi0_base` init, bfloat16, `num_inference_steps=10`).

Dataset: `hanapasta/pick_hold_v4s_train` — the **480-episode train split**
(240 act / 120 refuse_hazard / 120 refuse_ungrounded, 298 294 frames, 12
tasks). The full collection is 730 episodes = 480 train + 160 eval + 90
recover; only the train split was uploaded to the cluster and only it was used
here. **All probe and behaviour numbers are therefore measured on episodes the
policy was trained on.** Cross-validation prevents a direction from being
fitted and scored on the same episodes, but it does not make these held-out
data for the *policy*. The 160-episode eval split is the correct next target.

## Two code directories, deliberately

`code/` is the **as-run record** for stages 1–3 and is not edited. `code2/`
holds the stages 4–6 code written afterwards. Filenames are disjoint, so
nothing in `code/` was overwritten:

| directory | files |
|---|---|
| `code/` | `probe_extract_pi0.py`, `direction_controls.py`, `probe_cv.py`, `behaviour_check.py`, `ablate_sweeps.py` |
| `code2/` | `noise_paired_ablation.py`, `probe_extract_pinned.py`, `probe_cv_strict.py`, `channel_dissociation.py`, `steer_direction.py` |

Each new file states in its own docstring that it supersedes rather than
replaces its predecessor, and why.

## MANIFEST.md5 is stale for two files — this is expected, and here is the audit

`code/MANIFEST.md5` was written at 09-01 02:09. Two files were amended after
it, each to fix a defect found while the run was in progress:

| file | amended | fix |
|---|---|---|
| `behaviour_check.py` | 02:30 | `predict_action_chunk` returns the **raw normalised** model output; unnormalising is a separate postprocessor step. Without `ch = post(ch)` the policy's channels were in normalised units while the expert actions read from the parquet were in real units, so every per-channel comparison was meaningless. |
| `ablate_sweeps.py` | 03:43 | `probe_extract_pi0.py` stores `NL = n_blocks + 1` entries per stack (input to each block, plus input to the final norm), but only `n_blocks` are hookable. Iterating all `NL` raised `IndexError: index 18 is out of range`. Fixed for the **layer** path via `N_HOOKABLE`; the **rank** path was missed and still crashes (job 248198), which is why `sweep_rank_refusal_expert.json` does not exist. The rank path is fixed in `code2/noise_paired_ablation.py`. |

Every result reported post-dates the amendment of the file that produced it:

| result | written | code version | reported? |
|---|---|---|---|
| `behaviour_check.json` | 02:27 | **pre**-fix (normalised units) | **no — superseded** |
| `behaviour_check_unnorm.json` | 02:42 | post-fix | yes |
| `sweep_layer_*.json` (×4) | 04:35–06:55 | post-fix | retracted for a separate reason, below |
| `probe_cv.json`, `directions.npz` | 00:03 | unchanged, manifest OK | yes |

`MANIFEST.md5.as-of-0209` preserves the original hashes; `MANIFEST.md5` is
re-recorded against the code that actually produced the results.

## Stage 3 is retracted — the instrument, not the checkpoint

`sweep_layer_{refusal,band,random}_expert.json` and
`sweep_layer_refusal_trunk.json` are kept for the record but **must not be
read as ablation results**.

π0 is a flow-matching policy: `sample_actions()` draws noise from
`sample_noise()`, which is `torch.normal(0., 1., size=shape)` with no
generator and no seed. Every `predict_action_chunk` call therefore starts from
a fresh random draw. `ablate_sweeps.py` compares one clean call against one
intervened call, so its `collateral` statistic contains the policy's sampling
variance on top of any causal effect.

Measured directly (`code2/noise_paired_ablation.py --stage audit`, 30 act
episodes, **no hook installed at all**):

    two clean passes, fresh noise    0.5719, 0.6819, 0.5452   mean 0.600 ± 0.059
    two clean passes, pinned noise   0.000000  (bit-exact)
    flip statistic, fresh noise      -0.054

Against that floor, every stage-3 number is noise:

| arm | collateral | flip |
|---|---|---|
| refusal · expert | 0.618 | +0.015 |
| band · expert | 0.601 | +0.028 |
| random · expert | 0.587 | +0.031 |
| refusal · trunk | 0.644 | +0.008 |
| **noise floor (no intervention)** | **0.600 ± 0.059** | **−0.054** |

All four sit within one standard deviation of the no-intervention floor. The
`band` arm was the positive control — the most strongly encoded variable at
this checkpoint (CV AUROC 0.969) — and it moved behaviour no more than a
norm-matched random direction did. A positive control that fails is a
statement about the measurement, not about the thing measured, so stage 3
supports **no conclusion in either direction** about refusal.

Stages 4–6 pin the noise (`noise=` is accepted by `sample_actions` and
forwarded by `predict_action_chunk`), which makes the pass bit-exact and the
comparison exactly paired.

## Standing lesson

Run the null–null audit — two clean passes, no intervention — *before*
spending GPU on any intervention sweep. It costs minutes and it is the only
thing that distinguishes "no effect" from "no sensitivity".
