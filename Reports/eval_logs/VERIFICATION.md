# Ledger verification table — 2026-08-30

Every number on the Results Ledger recomputed from the raw logs in this
directory (`verify_ledger.py`, committed in `harness_audit/`). Frozen
protocol: `eval_frozen_v1.py` sha `a8cc240ba921`. Statuses: VERIFIED (matches as
printed), CORRECTED (page updated to the recomputed value), MEASURED
(previously inferred, now measured), UNVERIFIED (no committed source at
check time — resolved below or removed from the page).

Corrections applied to the rebuilt page:
- best-miss values now carry one decimal (53.1 / 205.4 / 73.0 / 56.6 mm);
- the 60k pick-up was **episode 3** of 20, not 4;
- nose-on-bearing slope is **+0.43** at the jaws-closest tick (the earlier
  +0.39 used the body-closest tick; definition now fixed and stated);
- the "undershoot collapsed because it was heading-derived" claim is
  RETRACTED: the collapse is real (+120 mm -> +1 mm under the pin) but an
  18 deg heading error on a 0.2 m lever explains ~10 mm, not 120 —
  mechanism recorded as unexplained;
- camera pixel tables now have a committed source
  (`harness_audit/camera_pixels.py` -> `camera_pixels.log`, 532 frames)
  and the page uses those refined values;
- oracle-yaw v2 trajectory rows verified against `eval_traj_v1.jsonl`
  (the local `eval_traj.jsonl` snapshot predated that run).

```
VERIFIED               | attr broken harness: success                      | page:               0/20 | recomputed:               0/20 | ladder_naive.log:(2, 21)
CORRECTED              | attr broken harness: best miss mm                 | page:                 53 | recomputed:               53.1 | ladder_naive.log
VERIFIED               | attr broken harness: median miss mm (new on page) | page:                  - | recomputed:              320.3 | ladder_naive.log
VERIFIED               | attr --startfix: success                          | page:               0/20 | recomputed:               0/20 | startfix_naive.log:(2, 21)
CORRECTED              | attr --startfix: best miss mm                     | page:                205 | recomputed:              205.4 | startfix_naive.log
VERIFIED               | attr --startfix: median miss mm (new on page)     | page:                  - | recomputed:              335.7 | startfix_naive.log
VERIFIED               | attr --platfix: success                           | page:               0/20 | recomputed:               0/20 | pfix_naive.log:(2, 21)
CORRECTED              | attr --platfix: best miss mm                      | page:                 73 | recomputed:               73.0 | pfix_naive.log
VERIFIED               | attr --platfix: median miss mm (new on page)      | page:                  - | recomputed:              338.4 | pfix_naive.log
VERIFIED               | attr both: success                                | page:               0/20 | recomputed:               0/20 | both_naive.log:(2, 21)
CORRECTED              | attr both: best miss mm                           | page:                 57 | recomputed:               56.6 | both_naive.log
VERIFIED               | attr both: median miss mm (new on page)           | page:                  - | recomputed:              270.4 | both_naive.log
VERIFIED               | 60k naive pick success                            | page:               0/20 | recomputed:               0/20 | b60k_naive.log
VERIFIED               | 60k naive picked-up                               | page:                  1 | recomputed:                  1 | b60k_naive.log
VERIFIED               | 60k naive best miss mm                            | page:                7.6 | recomputed:                7.6 | b60k_naive.log
CORRECTED              | 60k pick-up episode index                         | page:                  4 | recomputed:                  3 | b60k_naive.log
VERIFIED               | 60k pick-up object                                | page:            penguin | recomputed:      plush penguin | b60k_naive.log
VERIFIED               | 60k rtc pick success                              | page:               0/20 | recomputed:               0/20 | 60krtc_naive.log
VERIFIED               | 60k rtc best miss mm                              | page:               49.8 | recomputed:               49.8 | 60krtc_naive.log
VERIFIED               | 60k guided pick success                           | page:               0/20 | recomputed:               0/20 | 60kguided_naive.log
VERIFIED               | 60k guided best miss mm                           | page:               33.0 | recomputed:               33.0 | 60kguided_naive.log
VERIFIED               | 60k nav naive                                     | page:              12/20 | recomputed:              12/20 | lnav_naive_out.log
VERIFIED               | 60k comp naive                                    | page:               0/20 | recomputed:               0/20 | lnav_naive_out.log
VERIFIED               | 60k nav rtc                                       | page:               5/20 | recomputed:               5/20 | lnav_rtc_out.log
VERIFIED               | 60k comp rtc                                      | page:               0/20 | recomputed:               0/20 | lnav_rtc_out.log
VERIFIED               | 60k nav guided                                    | page:               7/20 | recomputed:               7/20 | lnav_guided_out.log
VERIFIED               | 60k comp guided                                   | page:               0/20 | recomputed:               0/20 | lnav_guided_out.log
VERIFIED               | 60k objyaw0 pick                                  | page:               0/20 | recomputed:               0/20 | objyaw_out.log
VERIFIED               | horizon10 picked                                  | page:                  0 | recomputed:                  0 | horizon10_out.log
VERIFIED               | horizon10 median mm                               | page:                412 | recomputed:                412 | horizon10_out.log
VERIFIED               | horizon10 IQR mm                                  | page:            286-636 | recomputed:            286-636 | horizon10_out.log
VERIFIED               | oracleyaw v1 picked                               | page:                  5 | recomputed:                  5 | oracleyaw_out.log
VERIFIED               | oracleyaw v1 median mm                            | page:               27.0 | recomputed:               27.0 | oracleyaw_out.log
VERIFIED               | oracleyaw v1 IQR                                  | page:           7.7-45.5 | recomputed:           7.7-45.5 | oracleyaw_out.log
VERIFIED               | oracleyaw v1 max mm                               | page:               65.1 | recomputed:               65.1 | oracleyaw_out.log
VERIFIED               | orayaw2 picked                                    | page:                  1 | recomputed:                  1 | orayaw2_out.log
VERIFIED               | orayaw2 median mm                                 | page:               41.1 | recomputed:               41.1 | orayaw2_out.log
VERIFIED               | paired v2-v1 median diff mm                       | page:                 +7 | recomputed:               +7.1 | both oracle logs
VERIFIED               | oracle-yaw pooled pick-up                         | page:               6/40 | recomputed:               6/40 | both oracle logs
VERIFIED               | audit n (tag=audit60k)                            | page:                 20 | recomputed:                 20 | eval_traj.jsonl
VERIFIED               | audit jaws median mm                              | page:                223 | recomputed:                223 | eval_traj.jsonl
VERIFIED               | audit body median mm                              | page:                365 | recomputed:                365 | eval_traj.jsonl
VERIFIED               | audit x_b sd mm                                   | page:                263 | recomputed:                263 | eval_traj.jsonl
VERIFIED               | audit x_b side split                              | page:              55/45 | recomputed:              55/45 | eval_traj.jsonl
VERIFIED               | audit y_b mean mm (undershoot)                    | page:               +120 | recomputed:               +120 | eval_traj.jsonl
VERIFIED               | audit y_b consistent frac                         | page:              18/20 | recomputed:              18/20 | eval_traj.jsonl
VERIFIED               | audit spawn corr r (n=20)                         | page:              +0.33 | recomputed:              +0.33 | eval_traj.jsonl
CORRECTED              | nose-on-bearing slope (closest)                   | page:              +0.39 | recomputed:              +0.43 | eval_traj.jsonl
VERIFIED               | object in nose FOV (median frac)                  | page:               1.00 | recomputed:               1.00 | eval_traj.jsonl
VERIFIED               | heading-sign match count                          | page:               2/20 | recomputed:               2/20 | eval_traj.jsonl
INFERRED->MEASURED     | undershoot y_b under pin (oracleyaw) mm           | page: (claimed 'collapsed') | recomputed:         +1 (sd 10) | eval_traj.jsonl
UNVERIFIED             | undershoot y_b under pin (orayaw2)                | page:                  ? | recomputed:  NO TAGGED RECORDS | eval_traj.jsonl
VERIFIED               | powered n                                         | page:                 66 | recomputed:                 66 | eval_traj_pre_audit.jsonl
VERIFIED               | powered radial r (demeaned)                       | page:              +0.44 | recomputed:              +0.44 | eval_traj_pre_audit.jsonl
VERIFIED               | powered x-slope (miss_x vs dev_x)                 | page:              -0.86 | recomputed:              -0.86 | eval_traj_pre_audit.jsonl
VERIFIED               | probe30k endpoint line                            | page:   2.90cm / 28.56cm | recomputed: 5s integrated endpoint err (m): 0.0290  | gt endpoint magnitude (m): 0.2856 | probe.log:24
VERIFIED               | probe60k endpoint line                            | page:             2.57cm | recomputed: 5s integrated endpoint err (m): 0.0257  | gt endpoint magnitude (m): 0.2856 | probe60k.log:24
VERIFIED               | fixfit5 verdict                                   | page:        A=3/3 B=3/3 | recomputed: PLATFORM-FIX5 A=3/3 B=3/3 | fixfit5.log:36
VERIFIED               | fixfit5 placement range mm                        | page:               7-13 | recomputed:               7-13 | fixfit5.log
VERIFIED               | repdiag2 replay misses mm                         | page:            102-152 | recomputed:            102-152 | repdiag2.log

60 claims checked; 52 verified as printed

MEASURED               | undershoot y_b under pin (oracleyaw) mm | page: (was 'collapsed, heading-derived') | recomputed: +1 (sd 10, n=20) | eval_traj_v1.jsonl
MEASURED               | undershoot y_b under pin (orayaw2) mm | page: (was 'collapsed, heading-derived') | recomputed: +52 (sd 123, n=20) | eval_traj_v1.jsonl
```
