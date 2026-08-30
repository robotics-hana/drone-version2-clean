# Harness audit & mechanism experiments — script/log/result map

Every claim in D46–D48 (`Reports/AIRVLA_RECREATION.md`) and in
`Reports/HARNESS_VALIDATION_DRAFT.md` traces to one script here and one raw
log in `Reports/eval_logs/`. The frozen evaluation protocol is
`../eval_frozen_v1.py` (sha `a8cc240ba921`) — torch-seeded, and it prints a
`PROV` provenance line (script sha, checkpoint, argv, seeds, timestamp) as
the first line of every results file. `../eval_pi0.py` is the working copy
where new diagnostics land; frozen results are only produced by the frozen
file.

Seeds used throughout: runner/scene seed **77000** (every eval run draws the
same spawn sequence, so cross-condition comparisons are paired), torch seeds
**1000 / 2000** (flow-matching sampling), dataset holdout split seed **0**
(`Reports/eval_logs/holdout_split.json`), collection seed family 91000.

| # | Experiment | Question | Script | Raw log(s) |
|---|---|---|---|---|
| 1 | Teacher-forced probe | Did the model learn the training mapping? (no simulator) | `open_loop_probe.py` | `probe.log` (30k), `probe60k.log` |
| 2 | Ground-truth replay, scene-faithful | Can the expert's own actions succeed through the eval harness? | `replay_diag2.py` | `repdiag2.log` (also `replay.log`, `repdiag.log` — earlier, RNG-unfaithful, struck) |
| 3 | Platform repair + acceptance | Fix the harness; accept only when replay passes | `fixfit5.py` (iterations 1–4 in `fixfit*.log`) | `fixfit5.log` |
| 4 | Attribution square | Which harness defect caused 0/20? (30k, one arm per fix) | `../eval_pi0.py` flags `--startfix` / `--platfix` | `startfix_naive.log`, `pfix_naive.log`, `both_naive.log` |
| 5 | Honest ladder, 60k | Task rates on the repaired harness | `../eval_pi0.py` | `b60k_naive.log`, `60krtc_naive.log`, `60kguided_naive.log`, `lnav_*_out.log`, `objyaw_out.log` |
| 6 | Spawn correlation (powered) | Does the miss grow with the object's distance from the training prior? | `powered_corr.py` | `eval_traj_pre_audit.jsonl` |
| 7 | Instrumented audit | Jaws-frame miss, body-frame offsets, heading checks | `audit_analysis.py` | `audit60_out.log`, `eval_traj.jsonl` (tag `audit60k`) |
| 8 | Horizon intervention | Is the 5 s open-loop chunk the limiter? (`--exech 10`) | `../eval_pi0.py` | `horizon10_out.log` |
| 9 | Oracle-yaw intervention | Does pinning heading at the true bearing fix the miss? (`--oracleyaw`, v2 releases on weld) | `../eval_pi0.py` | `oracleyaw_out.log`, `orayaw2_out.log` |
| 10 | Oracle-vector ablation | Upper bound: policy handed the exact jaws-to-target vector (+ corrupted-vector attendance control) | `build_oracle.py`, `../eval_pi0.py` `--oraclestate` / `--oraclecorrupt` | `oraeval_out.log`, `oracorrupt_out.log` (running) |
| 11 | Frozen n=60 baseline | Citable miss distribution + reproducible success footage | `../eval_frozen_v1.py` | `frozen60k_n60_out.log` (running) |

Checkpoints: `hanapasta/airvla_pi0_30k`, `hanapasta/airvla_pi0_60k` (private
HF; 90k pushed on completion). Training dataset `hanapasta/airvla_full`
(400 episodes); oracle variant `hanapasta/airvla_oracle` (state 10→13, built
by forward kinematics from the scene-state sidecar — no recollection).
