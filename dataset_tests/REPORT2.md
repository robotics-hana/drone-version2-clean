# Night tests 2: hedging, discrimination, parking window

Three free tests (no GPU, no training, no queue use), run 2026-08-31 on
the local machine. Scripts: `hedging_probe.py`, `discrimination_render.py`,
`parking_window.py`. Companion to `REPORT.md` (the reversal and
position-shortcut tests) and input to the v2 collection design.

## Pre-registered decision rules

Stated before the numbers (also printed by each script ahead of results):

- **Test A (hedging)**: if the midpoint of the two candidates is the
  closest of {target, midpoint, distractor} to the final gripper position
  in more than half the frozen-baseline episodes, selection failure is
  confirmed as the dominant lateral error and v2 must attack it directly.
- **Test B (discrimination)**: if the two objects are not separable in
  any camera stream at 1.0–1.5 m, the camera3 re-frame is justified — but
  for *discrimination*, not localisation, which is a different claim from
  the earlier D47/D49 one and must be recorded as such.
- **Test C (parking window)**: measurement only — the exact terminal
  window the v2 expert has to change.

---

## Test A — Is the policy hedging between the two objects?

**The rule does not fire: the midpoint is closest in 14/60, not >30.**

Method note: the distractor position is in no log, but it is a
deterministic function of the harness RNG stream. The stream was replayed
locally (Runner seed 77000, exact sampling sequence of the frozen run) and
validated against the logged **bin** positions — the one sampled quantity
that cannot physically move: **60/60 reproduce to <1 mm**, so the
reconstructed scenes are bit-faithful.

| Distance of final gripper position to… | median |
|---|---|
| target (spawn) | 281 mm |
| midpoint of the pair | 226 mm |
| distractor (spawn) | 414 mm |

Closest of the three: **target 32/60, midpoint 14/60, distractor 14/60.**
On the target→distractor axis (0 = target, 0.5 = midpoint, 1 =
distractor) the final position sits at median **0.22**, IQR [−0.11, 0.61]
— a lean toward the distractor side of the target, with broad scatter,
including a quarter of episodes ending on the *far* side of the target.
Per object: weight targets 0.18, penguin targets 0.25 — same shape.

**Reading**: pure split-the-difference hedging is not the dominant
terminal mode. The signature is broad terminal scatter biased ~20% of the
way toward the distractor — consistent with weak (not absent) selection
plus the terminal parking deficit, and with the earlier finding that the
final 10 s of every training label commands zero motion. Selection
remains *a* contributor (the lean is real; the prompt is inert; language
was the only cue in the data), but this test does not support making it
the sole target of v2 at the expense of the terminal-approach fix.

**Secondary finding**: the task object's end-of-episode position differs
from its spawn by >20 mm in **46/60 episodes** (max 1.14 m) — the policy
physically disturbs the scene in three quarters of episodes. Terminal
behaviour is not gentle hovering; it blunders through the workspace.

---

## Test B — Can the objects be told apart at 224 px?

**At the 1.0–1.5 m commitment band, no stream separates them cleanly:
the rule fires for a camera3 re-frame, recorded as a discrimination
claim, not the earlier localisation claim.**

Scene: trained geometry (objects side by side, 0.50 m apart), three
distance bands, three cameras, 224 px, segmentation-counted pixels and
mean object colour (`discrimination_crops.png` for the frames):

| Band | camera3 | camera2 (nose) | camera1 (wrist) |
|---|---|---|---|
| 1.0–1.5 m | 2 vs 5 px — **not separable** | 7 vs 12 px, ΔRGB 39 < pooled σ 43 — marginal | 26 vs 52 px, ΔRGB 6.6 — colour-identical, area 2× |
| 0.6–1.0 m | 2 vs 5 px — not separable | 15 vs 29 px, ΔRGB 19 — marginal | 64 vs 137 px, ΔRGB 3.1 — colour-identical |
| 0.4–0.6 m | 2 vs 5 px — not separable | 61 vs 107 px, distinct silhouettes | 265 vs 512 px, silhouette-separable |

Both objects render as near-black blobs (all mean RGB ≈ 40–90): colour
never separates them anywhere; silhouette/area separates them only below
~0.6 m in the drone-mounted streams. camera3, the fixed room overview,
never exceeds 5 px regardless of drone position — it cannot support
selection at all.

**Reading**: at the distance where the policy commits laterally, the
observation streams do not carry which-object-is-which. Even a perfectly
language-grounded policy would have to defer selection until ~0.6 m or
guess. This justifies the camera3 re-frame for **discrimination** — and
suggests v2's external camera must resolve the two objects at any drone
position, since it is the only stream whose resolution of the pair does
not depend on the drone getting close first.

---

## Test C — The parking window, precisely

For every pick episode: ticks between the last non-zero commanded motion
and the grip close (10 ticks = 1 s):

| Channel | median | IQR | max |
|---|---|---|---|
| horizontal, any motion | **107** | 106–111 | 117 |
| horizontal, >1 mm/tick | 108 | 107–111 | 117 |
| vertical, any motion | 252 | 117–270 | 387 |
| vertical, >1 mm/tick | 319 | 300–348 | 459 |

Minimum across all 225 episodes: **104 ticks (10.4 s)**. The number to
quote: *every* demonstration commands zero horizontal motion for the
final ~10.7 s — more than two full 50-step action chunks — before the
grip closes. The v2 expert's terminal approach must keep the commanded
motion non-zero into the close; this is the figure it is measured
against.

---

## Joint implication for the v2 design

Test C hands the primary change (continuous terminal approach) its exact
specification. Test B fires the camera3 re-frame under the corrected,
discrimination claim, and adds a requirement: the new external view must
resolve the object pair from every drone position. Test A moderates the
selection story — the v2 design should attack selection (corrective
episodes, discriminable views, language-necessary data are all still
motivated) but not at the expense of the terminal-approach fix, which the
hedging distribution says is co-dominant. The 46/60 scene-disturbance
figure adds an evaluation note: v2's eval should log object displacement
so "gentle" vs "blundering" approaches are distinguishable in numbers.
