# Dissertation Gaps Checklist (verified against the ledger 2026-09-10)

Status of every open item from the draft review, checked against
`finaldroneresults.md` / `V2_STUDY.md`. Three "pending" items were
already complete — the draft's snapshot was stale.

| # | Item | Status | Copy-ready detail |
|---|---|---|---|
| 1 | Plan D scaled-pairs fine-tune | IN FLIGHT | Collections close 2026-09-10 (~285 pairs projected); train overnight; results next day. |
| 2 | Learned-servo full n=60 | **DONE** | e5cfull (job 305965): 36/60 grasped (60.0%), 27/60 placed (45.0%), median 12.8 mm, nav 11/20 — beats scripted (24/22). The failed 10-ep pilot = e5cmini #1 (give-back-clock contract bug, diagnosed + fixed; mini #2 matched scripted 6/10). |
| 3 | Payload-compensation ablation | **DONE (falsification)** | Trim-on vs trim-off: millimetre-identical trajectories — the FF is inert in the PD stack (orphaned by the MPPI→PD migration). Report as the ablation finding per the reworded §5.4. |
| 4 | Inference latency (Table 13.1) | NOT MEASURED — cheap | Standalone bench (median/p95 of predict_action_chunk ×100) ≈ 30 min GPU; queue on request. |
| 5 | Carry-trajectory metrics for PURE configs (Table 11.1) | **FLAGGED: not meaningful** | Pure 047500 and C-15000 completed ONE carry each in 60 eps — no distribution exists. Restate the table over hybrid/E5C runs (22–27 carries; needs a state-logging re-run) or use expert-replay carry metrics. |
| 6 | Tucker et al. reference numbers (Table 3.1) | NEEDS THE PAPER | Must be transcribed from the source PDF; never from memory. |
| 7 | Ground-truth replay figures (§1.1) | **AVAILABLE** | Broken harness: deploy at tick ~5 (vs expert ~110), weld gate unsatisfiable → 0/3. Repaired: 3/3 weld at exact ticks 292/195/241, lifts +223/+475/+262 mm, clean release. Pre-fix policy numbers void (mini30k, mini30k_r2). |
| 8 | Excluded conditions (compositional / OOD / ACT+Diffusion) | CORRECTLY EXCLUDED | Pre-registered exclusions; frame compositional as a deliberate training hold-out (future eval-only probe). |

Flagged as not worth spending on: #5 as specified (n=1), #6 without
the PDF. Quick win available: #4 (one benchmark job).
