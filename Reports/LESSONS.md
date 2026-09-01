# Lessons for a redo — and the map of where everything is recorded

Written 2026-09-01, after the AirVLA recreation's first full cycle
(collection → training → harness audit → mechanism experiments → dataset
tests → v2 design). Two purposes: the ranked lessons a redo should build
in from day one, and the document map + corrections index so nobody
re-derives (or re-trusts) something this project already settled.

---

## Part 1 — The eight lessons, ranked by pain saved

1. **Validate the instrument before trusting any measurement through
   it.** Weeks of "0/20" results measured a broken harness, not the
   policy. Before the first policy eval: ground-truth replay (expert's
   own actions through the rig) and teacher-forced probe must both pass;
   the replay re-runs as a gate after every harness change. Companion
   rule from the interpretability side: run the null–null audit (two
   clean passes, no intervention) before any intervention sweep —
   unseeded flow-matching sampling variance swamped an entire ablation
   stage before it was measured.

2. **Design the expert with the learner in mind, and audit the data
   before training on it.** The five dataset tests (`dataset_tests/`)
   cost nothing and would have caught, pre-training: the expert parking
   motionless for 10.7 s (→ the policy learned to freeze), zero recovery
   demonstrations (the expert never arrives wrong), and cameras that
   cannot distinguish the two objects at commitment range (→ the
   language the data required was unlearnable from the pixels). For
   every skill the policy must show, verify the data *requires* it and
   the observations *support* it — as pre-training gates.

3. **Match training and evaluation distributions by construction.**
   Train/eval start-position overlap was 0/225 and nobody had checked.
   Define the eval distribution first; assert coverage numerically
   before collecting.

4. **Reproducibility and logging from day zero, sized for questions
   you'll ask later.** Torch seeding (flow sampling is unseeded by
   default), PROV lines (script sha + argv + seeds before any result),
   frozen protocol copies, never reuse a tag or filename. Retrofitting
   cost us the first-grasp footage (overwritten file) and a silently
   ignored flag (caught only because PROV recorded argv). Log full
   scene state per tick and video every episode: carry duration was
   uncomputable (no weld-state log) and the hedging test needed
   RNG-replay archaeology (distractor position unlogged).

5. **Holdout and metric semantics on day one.** The 90k result carries
   a permanent memorisation asterisk because no holdout existed when it
   trained. `picked` as a one-tick latch admitted grab-and-drops —
   caught by watching footage, which is itself the rule: check every
   scoring gate against video before quoting its number.

6. **Pre-registration, controls, statistics as the default.** Decision
   rules before numbers; an attendance/null control beside every oracle
   (the corrupted-vector arm is the only reason the oracle-vector
   result is interpretable); distances primary; rates only at n≥40 with
   CIs; scenes paired across seeds.

7. **Plan for attribution.** v1→v2 changes several things at once and
   must be reported as two studies. The sidecar re-render trick (one
   collection → two datasets differing in exactly one camera) is the
   pattern: design collections so single factors can be isolated later.

8. **The unglamorous infrastructure list.** Pin the physics version
   everywhere; never copy whole files to the cluster (surgical edits
   only — the solref clobber); verify bytes at the destination (CRLF
   kills Linux job scripts); GPU nodes for anything constructing a
   Renderer (EGL); one canonical scene-randomisation function shared by
   eval, probes and demos; a scene-disturbance metric in eval from the
   start (the policy moves the object in 46/60 episodes — discovered by
   accident).

**Meta-lesson**: nearly every week-long detour was an unverified
assumption a one-hour measurement would have killed. Measure the
assumption the day you make it.

---

## Part 2 — Document map: read these before redoing anything

| File | What it holds | Read it before… |
|---|---|---|
| `Reports/AIRVLA_RECREATION.md` | The decision log (D1–D50+): every design decision, experiment outcome, retraction and correction, in order, with dates. **The authoritative record.** | changing any design decision — the reason it is the way it is lives here |
| `Reports/EXPERIMENT_CHRONICLE.md` | The whole pipeline in narrative order — every python file, every experiment in plain sentences (aim / expectation / result). | onboarding anyone (including a future agent) to the project |
| `Reports/TEST_FILES.md` | One-line catalogue of every test script + the **metric definitions table** (incl. the `picked` latch caveat). | quoting any metric or re-running any script |
| `Reports/eval_logs/VERIFICATION.md` | The 60-claim verification pass (52 verified, 8 corrected) tying every ledger number to its raw log. | trusting any number not in a raw log |
| `Reports/results_ledger.html` | The verified results, one tab per experiment, template-structured, confounds marked. | comparing new results to old ones |
| `Reports/HARNESS_VALIDATION_DRAFT.md` | Dissertation-section draft of the harness audit + methods note (distances not rates). | writing up |
| `dataset_tests/REPORT.md`, `REPORT2.md` | The five pre-registered dataset tests with decision rules stated before numbers. | proposing any new data collection |
| `v2_design/CONFIRMATION.md` | The v2 collection spec: camera pose (measured), spawn envelope, terminal law, corrective episodes, composition — each change traced to a measurement. | collecting v2 (requires sign-off first) |
| `Sim'n'Real/Mujoco/harness_audit/README.md` | Experiment → script → log map and the seed inventory. | re-running any audit-era analysis |
| `MYRIAD_AGENT_GUIDE.md` | Cluster rules, queue etiquette, login-node limits, the queue-behind snippet. | touching Myriad at all |
| `Reports/LESSONS.md` | This file. | starting the redo |

## Part 3 — Corrections index: things once believed that are WRONG

A redo that reads only early documents would re-absorb these. Each is
struck/corrected at source; this is the quick list.

| Once believed | Corrected to | Where |
|---|---|---|
| Early ladder results (0/20 everywhere) reflect the policy | They reflected five harness defects; invalid | D46, `HARNESS_VALIDATION_DRAFT.md` |
| Ground-truth replay v1 was a valid test | v1 re-drew ~11 RNG values on reset; only the v2 snapshot replay counts | D46 |
| "Yaw contributes ~45% of the miss" (slope-gap argument) | Retracted — the jaw-vs-body slope gap is a kinematic identity, not aiming evidence | D47 |
| The 18/20 heading-opposition test falsifies/supports yaw-following | Non-discriminating — prior-following predicts the same; only the oracle-yaw *intervention* discriminates | D47 |
| Undershoot is heading-derived | Retracted (18° on a 0.2 m lever ≈ 1 cm, not 12 cm); collapse measured, mechanism unexplained | D47 |
| First grasp was episode 4 | Episode 3 (and its footage is lost — the preservation rules date from this) | `VERIFICATION.md` |
| Seed-2000 oracle-yaw ran with its flag | Flag silently ignored (post-dated the v1 freeze); run banked as a naive baseline; treatment re-run on frozen v2 | D48-OUTCOMES-2 |
| "Training always put the prompted object at the task spot" (D50 mechanism) | **Falsified by measurement**: best position rule 52.3% = chance; language was the only cue and the model failed to learn it | D50 correction, `dataset_tests/REPORT.md` |
| "3/60 picked, all on video" (read as three pickups) | 2 sustained carries + 1 momentary grab — `picked` latches on one tick of weld + 12 cm lift | D48-OUTCOMES-2 semantics note, `TEST_FILES.md` |
| Selection failure is the dominant terminal error | Hedging rule did not fire (midpoint closest 14/60); parking deficit is co-dominant | `dataset_tests/REPORT2.md` |
| Camera3 re-frame is about localisation | The measured justification is **discrimination** (2–5 px, colour never separates) — a different claim | `dataset_tests/REPORT2.md`, D49 vs `v2_design/` |
| A "keep the distractor central" spawn rule would be harmless | It would leak position at ~79%, recreating the confound; v2 uses a symmetric envelope + coin-flip assignment | `v2_design/CONFIRMATION.md` |
