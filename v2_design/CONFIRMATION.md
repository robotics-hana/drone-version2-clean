# v2 collection — sign-off package

> **AMENDED 2026-09-01 per Hana's design decisions**: (1) the goal box
> sits **next to the table** — this replaces §2's taped square AND the
> far-spawned bin; (2) camera3 is the measured workspace camera of §1,
> confirmed; (3) the **penguin is blue** (fixes Test B's finding that
> colour never separated the objects — both rendered near-black);
> (4) the terminal approach is one continuous slow forward creep with
> the grip closing in motion — **no pauses, no press-retreat-repress**;
> (5) every leg is **hover → yaw in place → translate nose-first**, so
> the heading channel (the measured bottleneck; non-zero in only 13%
> of v1 frames) is deliberately exercised toward the object and again
> toward the box. Implemented in `Sim'n'Real/Mujoco/collect_v2.py`
> (new file; imports, never edits, `collect_airvla.py`; camera3 renders
> as a free camera so no XML the frozen eval loads is touched; the
> penguin recolour is applied at model load, not in the v1 XML).
> Success criterion returns to "object inside the box, weld released"
> (the box is back). Demo episodes: `Reports/v2_demo.mp4`.
>
> **DEMO VERDICT (Sparks job 935, mujoco 3.11)**: all three episodes
> pass — weight standard 35.3 mm into the box, penguin standard
> 17.9 mm, penguin corrective 22.7 mm; grasp fired in motion in every
> episode (closest jaw approach 1.2–1.9 mm) and the parking window
> measured **1 tick vs v1's 107**. Five defects were found and fixed
> during demo iteration, each traced in collect_v2.py's comments:
> ramp-time overshoot on the weight stem, missing trim compensation on
> every loaded phase (3 m runaway, traced numerically), the release
> point ignoring the carry offset (193 mm systematic), table-edge leg
> collisions on diagonal approaches, and a pathological teleported
> corrective start (replaced by a flown displaced approach). Remaining
> pre-collection note: the shared Sparks tree's collect_demos.py has
> forked toward the other agent's study (1,672 lines) — the real
> collection runs from a clean checkout, not the shared tree.

Everything here requires confirmation **before any collection starts**.
Each design change is traceable to a measurement; the measurement is named
next to each. Scripts and images in this folder reproduce every number.

## 1. Camera3 pose (from REPORT2 Test B — a discrimination claim)

**Proposed**: fixed external camera on the room side, behind the flight
line, framing the task area — `pos (0.00, 2.25, 1.22), lookat (0.05, 0.08,
0.30), fovy 52°`. Rendered frames: `cam3v2_signoff.png` (five scenes,
labelled), `cam3v2_signoff_224.png` (exactly what the policy sees).

Measured pixel counts at 224 px, segmentation-counted, across the v2 spawn
envelope (`camera_pose.py`, all five scenes PASS):

| Scene | weight | penguin | ratio | tape | gate |
|---|---|---|---|---|---|
| pick centre | 34 px | 67 px | 2.0 | 138 px | — |
| pick far-left-front | 24 px | 44 px | 1.9 | 104 px | — |
| pick far-right-back | 55 px | 104 px | 1.9 | 118 px | — |
| nav gate-left | 34 px | 65 px | 1.9 | 164 px | 1692 px |
| nav gate-right | 46 px | 79 px | 1.7 | 164 px | 1710 px |

Contrast with today's camera3: **2–5 px at every distance** — the current
external view cannot support object selection at all.

Two measured facts shaped the design, and both become v2 spec:

- **A side (−x) camera fails**: the object pair is separated along x, so a
  side camera views it end-on and one object occludes the other (measured
  6–14 px). The camera must sit on the ±y axis.
- **The frame edge sets a spawn envelope**: beyond |x| ≈ 0.7–0.8 (depth-
  dependent) an object clips the frame and its apparent size inverts
  (measured: penguin 39 px < weight 46 px). Restricting only the
  *distractor* to the safe zone would leak position information — "the
  outer object is the target" would be right ≈79% of the time, recreating
  the confound. Therefore: **both objects sampled inside |x| ≤ 0.68
  (target sweep ±0.45), and target/distractor assigned by coin flip** —
  position uninformative by construction, re-verified by re-running
  `position_shortcut.py` on the collected data as a pre-training gate.

Known characteristic to accept: the drone crosses this view and can
partially occlude the scene in some frames (visible in the far-left-front
tile); the paper's external camera shares this property.

## 2. Taped-square goal (fidelity; replaces the bin)

25 cm outer, 3 cm yellow tape, flat on the table top (rendered in every
sign-off frame; 93–164 px from the new camera — the bin volume it
replaces is a 3D obstacle the drone had to clear). Placement sampled on
the table away from the object pair.

**Success criterion change**: from "object inside the bin volume, weld
released" to **"object centre inside the taped region on the table
surface (z at table height), weld released"**. The 18-gate quality
harness keeps every other gate unchanged.

## 3. Terminal approach (from Test C — the primary change)

Measured v1: **every** demonstration commands zero horizontal motion for
the final 104–117 ticks (median 107 ≈ 10.7 s, two full action chunks)
before the grip closes; the vertical channel is silent even longer. The
policy learned exactly this: park short, hold still.

**Proposed v2 expert law** (`terminal_profile.py`, plot
`terminal_profile.png`):

```
v(t) = clip(0.35 · d(t) / FPS,  3 mm/tick,  30 mm/tick)
grip fires at d ≤ 15 mm — while still moving
```

Commanded motion is non-zero at every tick through the close (floor
3 mm/tick); from 1.0 m the approach takes ~93 ticks and never stops. The
acceptance test for the new expert: `parking_window.py` on the collected
data must report a zero-tick parking window (last non-zero horizontal
command at, or within a tick of, the grip close) — plus the existing
18-gate harness.

## 4. Corrective episodes (from Test C / Fact 1)

The behaviour the policy has never seen: arriving wrong and fixing it.

- **Count**: 80 episodes ≈ 25% of the manipulation set.
- **Start state**: hover at grasp altitude, arm deployed, displaced from
  the target by **r ~ U(5, 20) cm at a uniform random bearing** — the
  5–20 cm band brackets the measured miss distribution's lower half
  (frozen baseline IQR 110–359 mm, oracle-yaw 22–59 mm).
- **Demonstration**: the same v2 terminal law from the displaced start —
  correct laterally, close in motion, lift. Short (~15 s), so these are
  cheap.
- Prompted normally (both objects present, coin-flip assignment), so they
  also carry language-grounding signal.

## 5. Start distribution (from the 0/225 overlap finding, D46)

Drone spawn altitude widened from the single computed deck altitude to
U(0.21, 0.60); lateral start region unchanged. Evaluation then no longer
needs `--startfix` narrowing to match a degenerate training distribution.

## 6. Dataset composition and holdout

| Slice | Episodes | Notes |
|---|---|---|
| standard picks (v2 expert) | 240 | coin-flip target assignment |
| corrective picks | 80 | §4 |
| nav (gate + hover) | 160 | unchanged design, new camera |
| **total** | **480** | v1 was 400 |
| holdout | 40 stratified | excluded via `dataset.episodes` at train time |

## 7. Camera attribution (one collection, two datasets)

Collected once with the new expert and scene; the scene-state sidecar then
renders a **second variant differing only in camera3** (old room-overview
pose vs new pose, identical episodes and actions). Two training runs from
one collection isolate the camera effect; everything else is v1-vs-v2 as
two studies, stated plainly, not a controlled ablation.

## Constraints honoured

`airvla_full` untouched (v1 record) · v1 checkpoints stay on HF ·
`eval_frozen_v1.py` unmodified, v2 evaluated on the same protocol and
seeds · new collector code in new files (`collect_v2.py`, importing, not
editing, `collect_airvla.py`) · v1 vs v2 reported as two studies.

## What sign-off means

Confirming: (a) the camera pose and its spawn-envelope consequence,
(b) the taped square and its success criterion, (c) the terminal law,
(d) the corrective-episode design, (e) the composition table. On
confirmation, `collect_v2.py` gets written against this spec, verified by
the 18-gate harness + the two pre-training gates named above, then
collected on Sparks (deterministic base, mujoco 3.11).
