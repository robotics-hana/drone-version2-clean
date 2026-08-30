# Draft §: Validating the Evaluation Instrument

*Dissertation section draft, 2026-08-30. Numbers are final; cross-references
to figures/sections to be filled in. Style: first person plural to match the
rest of the methods chapter.*

---

## Motivation

Closed-loop evaluation of an imitation policy runs the policy through an
*evaluation harness*: code that reconstructs, at test time, everything the
demonstration platform did at collection time — scene resets, actuator
limits, platform-owned automation (in our case, the arm state machine and
the grasp weld), and the distribution of initial conditions. The harness is
itself software, and it is software that no training-time signal ever
exercises: a defect in it produces no loss spike, no crash, and no error
message — only a policy that appears incompetent. Over 384 evaluation
episodes (the Phase E ladder and its dataset-v2 successor), our π₀ policy
scored 0/20 on the pick task in every configuration, a result we initially
attributed to the policy. It was instead the instrument.

We argue that two conditions are *necessary* before any closed-loop number
from such a harness is interpretable, and give one cheap test for each.

## Two necessary conditions, two tests

**Condition 1 — the policy must have learned the demonstrated mapping.**
Otherwise closed-loop failure is uninformative about the harness. The test
is a *teacher-forced probe*: feed stored training observations to the
policy and compare its predicted action chunks against the logged expert
actions. No simulator is involved, so closed-loop compounding cannot
contaminate the result. Our 30k-step checkpoint reproduced the training
action distribution almost exactly — per-dimension correlation +0.88 to
+0.999, predicted-to-ground-truth spread ratio 0.95–1.01, and 2.9 cm of
integrated endpoint error over a 5 s action chunk whose ground-truth
magnitude is 28.6 cm. This single measurement falsified the two most
tempting explanations for 0/20 — undertraining and collapse to the
marginal action — before any harness work began.

**Condition 2 — ground-truth actions must succeed through the harness.**
If the expert's own logged actions fail when replayed through the
evaluation pipeline, no policy can succeed in it, and every number it has
produced measures the harness. The test is a *ground-truth replay* on a
byte-faithful world: snapshot the complete initial state at the first
expert tick of a collection episode (generalised coordinates and
velocities, plus every model field the scene reset mutates — body
positions, orientations, masses — and the derived quantities scoring
depends on), restore it exactly, and drive the evaluation platform with
the recorded action sequence. Faithfulness matters: our first replay
re-invoked the scene-reset routine with captured arguments, but the reset
internally draws eleven values from the episode RNG (object yaw, bin
position, distractor placement), so the replay world silently differed
from the collection world and the first replay result had to be struck.

Our harness failed Condition 2 comprehensively: the expert's actions,
which succeed in the collection simulator, missed the grasp point by
102–152 mm in the evaluation harness, welded nothing, and in two of three
episodes the object crossed the room *with no drone contact ever logged*.

## What was wrong

The replay's instrumentation (per-tick automaton state, weld events,
contact pairs, aperture traces) localised five independent defects:

| # | Defect | Evidence | Repair |
|---|---|---|---|
| 1 | Arm automaton gated on a stale altitude constant (`sp_z < 0.42`) inherited from an earlier task geometry; the current expert approaches *and grasps* at z ≈ 0.53 | jaws parked 100–150 mm from the grasp point at every true grasp moment | automaton refit (below) |
| 2 | Grasp weld triggered on gripper aperture alone; jaws closing on air sweep through the trigger window, and the resulting "air-weld" rigidly tows the object at its current offset | object moved 2 m with zero drone-object contacts | weld requires pad–object **contact** with seated aperture — physically impossible on air |
| 3 | Evaluation start altitudes z ~ U(0.55, 0.95) versus collection's deck spawn z ∈ [0.21, 0.30] | **0/225** training pick episodes share any start altitude with any evaluation episode; the navigation task, whose overlap is 161/175, is exactly the task that worked (18/20 gate crossings) | evaluation start distribution replicated from the collector, including its rejection sampling |
| 4 | Action clip ±0.030 versus the collector's per-tick limit 0.035 | computed analytically from the dataset: ~2.3 cm median shortfall per episode — real but minor | clip 0.035 |
| 5 | Weld contact gate required *consecutive* contact ticks; contact with the plush object flickers tick-to-tick from solver jitter | pinch trace at 2.0 mm from aim: contact alternating T/F/T/F | debounce (2-of-last-4) |

Defect 1 had a second-order consequence we did not anticipate: the wrist
camera is mounted on the arm, so an automaton holding the wrong arm pose
also rendered one of the policy's three input streams from a configuration
that never occurs at that phase of training.

Repairing the automaton required abandoning its named pose constants
altogether. The pose named `q_grasp` in the platform code lies ~0.13 m of
forward kinematics away from the pose the expert actually holds at
release; we instead *calibrated* the three automaton poses (travel,
deploy, tuck) from the demonstrations themselves — the mean commanded arm
configuration over fixed windows before the weld and before the release —
which is offline system identification from training data, not a runtime
privilege. Acceptance was defined in advance: the repair is accepted only
when ground-truth replay places the object, never when the diff looks
right. The validated harness places all three replay episodes 7–13 mm from
the bin centre, with weld engagement within four ticks of the expert's.

## The audit: falsified versus unmeasured

Every conclusion previously derived from closed-loop rollouts was
re-classified once the harness failed Condition 2. A result upstream of a
broken instrument is not *false* — it is *unmeasured*:

| Prior conclusion | Basis | Status after audit |
|---|---|---|
| Undertraining / marginal collapse | teacher-forced probe | **falsified** (rollout-independent; stands) |
| Action-clip causal for the miss | dataset analytics | **excluded** (rollout-independent; stands) |
| Inference ladder flat (naive/RTC/guided) | rollouts | unmeasured → re-measured on the repaired harness |
| Guidance ineffective at all scales | rollouts | unmeasured → re-measured |
| World-frame action convention exonerated | rollouts | unmeasured → re-measured |
| "More demonstrations don't help" | rollouts | unmeasured (re-test requires retraining; noted as open) |

Re-measurement on the repaired harness gives the first interpretable
result: the policy still scores 0/20 on picks at 30k steps, so the harness
defects were *necessary but not sufficient* to explain the failure — but
at 60k steps the policy achieves its first genuine grasp (1/20 picked, with
air-welds now physically impossible) and a best approach of 7.6 mm, inside
the grasp window. The capability is emerging, and the remaining gap is
attributable to measured observation deficits (§ ref: the external camera
contributes 0–4 pixels of target object at the policy's input resolution).

## Remark

The two tests cost minutes of compute — the probe needs one GPU pass over
forty training frames; the replay needs three physics episodes — and
between them they arbitrate the central attribution question (policy
versus instrument) that 384 evaluation episodes could not. We suggest both
as preconditions for reporting closed-loop imitation results on any
platform where the evaluation harness re-implements collection-time
automation, and we note the general principle: an evaluation harness is
untested code on the load-bearing path of every empirical claim.

## Status appendix: the ongoing debug (2026-08-30, live)

*This section tracks the investigation as it runs and will be folded into
the results chapter once the confirmatory measurement lands.*

With the harness validated, the residual question became *why the policy
still misses by ~200 mm*. The debugging discipline established above was
applied to our own analyses, and caught us twice more:

1. A 26-episode trajectory analysis ("the drone stalls at the expert's
   standoff") was **discarded on review**: its episodes were selected by
   counting backwards through a shared log file (unverifiable — 307
   records share the same mode label), its distances were measured from
   the airframe rather than the jaws, and its "lateral bias" test
   measured sideways-of-a-world-line rather than sideways-in-the-body-
   frame, which is what was actually observed in the wrist video. The
   trajectory logger now records a per-run tag, jaw positions, and
   heading; a tagged 20-episode audit run re-measures all three
   quantities with the axis semantics written into the output (nose =
   body −y; x_b is the lateral axis).

2. The spawn-correlation test — does the miss grow with the object's
   distance from the training-set average position? — was run first at
   proper power (n = 66, all honest-harness 60k blocks, identified by
   deck-spawn altitude with asserted block counts rather than trusted
   ordering; the count assertion immediately caught a forgotten
   twenty-episode block). Result: radial Mahalanobis correlation
   r = +0.44 (significance threshold ≈ 0.24), and the directional
   version is unambiguous — **the x-component of the miss regresses on
   the object's x-deviation from the training mean with slope −0.86
   (r = −0.81), while the y-axis slope is +0.02**. The policy ignores
   the object's lateral position and flies to the prior; it tracks the
   approach axis. This is the predicted signature of the measured
   observation deficit (the external camera contributes 0–4 pixels of
   target; the wrist camera resolves it only in the final half-metre),
   and it initially pointed at the external-camera re-render as the gate.

3. The reviewer's heading check then falsified the *strong* form of that
   conclusion — the object remains inside the nose camera's ±65° field
   of view for a median 100% of every approach (20/20), so the lateral
   information **is** present in a stream the policy receives — and
   the same check, once its sign convention was verified synthetically,
   caught this report's own first reading. We had written the
   yaw-channel hypothesis off on "2 of 20 sign agreement"; the
   mechanism in fact predicts sign *opposition* (object bearing left →
   fly past on the right), so 2/20 equal is **18/20 opposite**
   (binomial p ≈ 2×10⁻⁴). The same reviewer round then showed the sign
   test cannot discriminate at all: a policy flying straight to a fixed
   prior point also points roughly at its destination, so heading error
   and miss direction derive from the same quantity and come out
   opposite under either hypothesis. The aiming channel is therefore
   *no longer excluded* rather than supported; the discriminating test
   is an intervention — pinning the commanded heading at the true
   bearing — which is queued. Decomposition on identical episodes:
   body-x tracks object-x at slope +0.22 and jaw-x at +0.40 — a gap
   that is a kinematic identity (any yaw swings the nose-mounted
   jaws laterally), not an attribution to the aiming channel. The
   direct measurement — nose angle regressed on bearing to object —
   gives slope +0.39 at closest approach and mid-approach alike:
   heading, body and jaws all close roughly 40% of the lateral
   offset, and the causal owner is decided by intervention, with
   the predicted outcome of each experiment registered in the
   decision log (D48) before any result landed. Two further measurements complete
   the picture: the nose camera resolves the target at ~11 px at
   0.9 m, ~100 px at 0.4 m, and 650–935 px at 0.2 m (marginal early,
   usable mid-approach); and the final jaw position tracks the
   object's lateral coordinate with slope +0.40 — the policy uses
   some of the signal and closes roughly 40% of the offset. The
   surviving mechanism is partial visual tracking plus systematic
   under-aiming, with control-frequency and slot effects unresolved.

**Current intervention order, evidence-gated (second revision):**
(1) *horizon test* — replan every ten actions instead of fifty, no
retraining; if lateral scatter collapses, the diagnosis is
control-frequency, not perception; (2) *oracle-yaw evaluation* — the
platform pins the commanded heading at the bearing to the object;
the direct test of the reinstated aiming channel; (3) *oracle-vector
ablation* — the true gripper-to-target vector appended to the state
(a derived column, a short fine-tune), bounding the problem from
above: reliable grasps mean everything downstream of perception
works; persistent lateral misses mean perception was never the
constraint — paired with an *attendance control* (the same policy
evaluated with the vector's lateral direction randomised per episode)
so that a null distinguishes "perception was not the constraint" from
"the fine-tune never learned to attend to the new feature". The rename-map swap arm continues on otherwise-idle
hardware but is demoted — it tests a soft prior at 38 hours' cost
and is confounded with the newly introduced holdout. The
external-camera re-render remains on its merits, off the critical
path.

---

*Supporting artefacts: `eval_pi0.py` flags `--platfix`, `--startfix`;
validation script `fixfit5.py` (acceptance A: expert arm, B: deployable
automaton — both 3/3); teacher-forced probe `open_loop_probe.py`;
spawn-correlation script `powered_corr.py`; decision log entries D46
(harness audit and repair) and D47 (lateral-blindness localisation) in
`AIRVLA_RECREATION.md`.*
