# v2 failure-review set — 10 pick episodes (frozen eval, ckpt 047500)

All from `full47k5` (n=60). What to watch for in each, with the
trajectory diagnosis (lateral = jaw-to-target horizontal distance;
the weld gate needs <10 mm lateral AND <15 mm vertical):

| Video | miss (3D) | What it shows |
|---|---|---|
| pick09 | 9.8 mm → **SUCCESS** | The one that worked: hovers ~17 s within 5 cm of the target (174 ticks) — long enough for the slow creep to align — welds at tick 699, carries, places in the box. |
| pick23 | 17.8 mm | Arrives well, but sits ~2 cm off laterally and ~4 cm HIGH; dwells only ~4 s then leaves for the box. |
| pick44 | 20.1 mm | Never gets under 3 cm laterally; ~10 cm LOW at closest pass (jaws near table height); moves on after ~4 s. |
| pick28 | 24.9 mm | **Laterally PERFECT (3 mm!)** but ~13 cm too low — pure height error; gives up after ~5 s. |
| pick41 | 25.9 mm | ~2 cm off and ~7 cm high; brief hover, then continues the routine. |
| pick50 | 27.6 mm | ~2 cm off at roughly the right height — the closest "should have converted" failure besides ep23. |
| pick00 | 73.0 mm | Typical mid-miss: approach is right-shaped but parks one object-width away, never enters the creep zone. |
| pick03 | 64.6 mm | Same family; also brushes the penguin once (obj_hits 1). |
| pick01 | 491.3 mm | Far-miss family: transits to the wrong region of the table, hovers there, proceeds to the box on schedule. |
| pick10 | 593.6 mm | The rough flight: 31 table-contact ticks + 22 object strikes — worst contact record in the run. |

## The two "obvious things going wrong"

1. **Short terminal dwell — the policy runs on a schedule, not on
   success.** Failures hover only ~4–5 s near the target (39–54
   ticks within 5 cm) then proceed to the box empty-handed; the one
   success dwelt ~17 s. Training data contains no episode where the
   approach fails — so the policy never learned to persist/retry,
   only the expert's timeline (grasp → leave). Fix: terminal-
   corrective data (E3) and/or closed-loop replanning (E1/E2).
2. **Small bi-axial terminal offsets it never servos out.** Near
   misses are 2–3 cm lateral and/or 4–13 cm vertical (both
   directions — ep28 is dead-centred but 13 cm low; ep23/41 are
   high). The expert removes exactly these offsets with its
   every-tick 3 mm/tick creep; the policy, committed to 50-step
   open-loop chunks, drifts through the window instead. Fix: E1
   (execute 10/50, replan at 1 Hz) and E2 (RTC).

Analysis source: `eval_logs/eval_v2_traj_full47k5.jsonl`, EVAL rows
in `v2full47k5.log`.
