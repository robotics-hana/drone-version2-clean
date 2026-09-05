# Dissertation evidence checklist — dataset & training minimum set

The figures/tables the dissertation must eventually contain (target: 90%).
Status legend: ✅ artifact exists · 🔶 producible now / in progress ·
⬜ awaiting a dependency. Every ✅/🔶 names its source so nothing is
asserted without a file behind it.

## Dataset

| Evidence | Status | Where / dependency |
|---|---|---|
| Dataset composition table | ✅ | `finaldroneresults.md` (270 std / 90 corr / 240 nav, object counts per flavour); raw: `eval_logs/v2_manifest_71000.jsonl` |
| Spatial distribution plot | 🔶 | scatter of target/distractor spawns from the manifest — produce with the existing `dataset_tests/position_shortcut.py`-style plot on v2 data (queued alongside the validation sweep) |
| Relevant variable distributions | ✅ table / 🔶 plots | min/max/mean/SD table for target x/y, distractor x/y, episode length in `finaldroneresults.md` (from `v2_dataset_stats.py`); histogram plots to accompany |
| Success/rejection statistics | ✅ | 600/600 banked, ZERO rejections (manifest records every attempt); v1 comparison: banking-gate discipline documented in `TEST_FILES.md` |
| Example demonstrations | ✅ | `Reports/v2_demo.mp4` (10 pick episodes), `Reports/v2_nav_demo.mp4` (6 nav), `Reports/v2_trial_cam3.mp4` (stored-data verbatim) |
| Train/validation/test methodology | ✅ | `finaldroneresults.md` pre-registration: 480/120 episode-level stratified split (`eval_logs/v2_split.json`), test = frozen simulator eval on disjoint seed families, leakage-prevention list |

## Training

| Evidence | Status | Where / dependency |
|---|---|---|
| Training loss curve | 🔶 | raw data in `v2train_resume.log` + `v2train60k.log` (tqdm loss per step) — plot pending |
| Validation loss curve | ✅ | `eval_logs/v2_valcurve_60k.json` — 24 checkpoints (2.5k–60k), 720 pinned-noise windows each, paired across checkpoints |
| Final checkpoint-selection criterion | ✅ applied | lowest validation action-MSE, frozen BEFORE results; selected **047500** (0.0000455) over the full 60k curve; never revised against test |
| Validation action error | ✅ | 047500: overall MSE 0.0000455 (`v2_valcurve_60k.json`) |
| Optional: action-dimension breakdown | ✅ | per_dim (7) + per_flavour (std/corr/nav) for every checkpoint from 15k on, incl. the winner |
| Optional: episode scaling experiment | ⬜ | 120/240/480-episode trainings, same protocol, `--dataset.episodes` sublists — queued after the main run |

## Appendix material (granular)

| Item | Status | Where |
|---|---|---|
| Full episode metadata | ✅ | `eval_logs/v2_manifest_71000.jsonl` (per-attempt: kind, object, coordinates, ticks, gate counters, outcome) |
| All hyperparameters | ✅ | `finaldroneresults.md` §T (frozen protocol) + the run log's config dump |
| Rejected episodes | ✅ | none in v2 (0/600) — the manifest proves it; v1-era rejections in the collection-review history |
| Complete seed list | ✅ | collection 71000 · trial 72000 · split 424242 · training 1000 · eval scene families 77000-series · probe 88000 · demos 31–33000 — inventory also in `harness_audit/README.md` |
| Individual trajectory data | ✅ | parquet in `hanapasta/airvla_v2` (states/actions per tick); eval trajectories in `eval_logs/eval_traj_v*.jsonl` |
| Extra training curves | ⬜ | with the training artifacts |
| Additional qualitative examples | ✅ | demo iteration videos + contact sheets in `Reports/` (grasp windows, endings, debug sheets) |

## Bookkeeping

- Dataset version pinned: `hanapasta/airvla_v2` revision `2e40a5c89746`.
- Base model pinned: `lerobot/pi0_base` snapshot `25c379b52ba2…`.
- Code version: this repo's commits (collector approved at `a313575`).
- Multi-seed training (T9) deliberately deferred (Hana, 2026-09-03) —
  revisit after the main results.
