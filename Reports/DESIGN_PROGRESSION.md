# From Overhead Reach to the Hybrid Policy: How the Demonstration,
# Training and Validation Design Evolved

This document narrates the project's design progression as a single
arc — each stage is described by what was tried, what was measured,
why it failed or fell short, and what the next stage changed in
response. Detailed numbers live in `V2_STUDY.md` and
`finaldroneresults.md`; this is the story that connects them.

## 1. The overhead reach: losing the object at the decisive moment

The earliest grasp choreography brought the vehicle above the target
and descended onto it. It was highly unsuccessful for a reason that
became a design principle: as the drone closed vertically, the object
fell out of the cameras' useful field of view exactly when precision
mattered most. A policy trained on such demonstrations would have to
perform its finest manoeuvre blind. The lesson — the demonstrator
must keep the object visible through the terminal phase — pushed the
design toward approaches that come in level, with the object held in
front of the wrist cameras all the way to the close.

## 2. The v1 side approach: a grasp at last, but the wrong habits and
## the wrong viewpoint

The first full campaign (v1) flew a side-on approach and produced a
working demonstrator and a policy that could occasionally touch
success: on the frozen sixty-episode evaluation the median
jaw-to-target miss was 242 mm, with three latched lifts and **no
completed place**. Two data measurements explained the shortfall
better than any single failure. First, the demonstrator hovered
motionless above the object for a median of 107 control steps before
closing — and the policy faithfully copied the parking, because the
policy copies the shape of trajectories, not their outcomes. Second,
the third camera was positioned so far back that it delivered a
laboratory overview rather than a workspace view: the pixels that
mattered for manipulation occupied a small fraction of the frame.
The v1 era ended with one honest grasp on film and a list of
properties the next dataset must not contain.

## 3. The v2 redesign: smooth motion and a camera that watches the
## work

v2 rebuilt the demonstration around one rule — the demonstrator
should never produce behaviour that would be unhelpful if imitated.
Stalling was eliminated end to end: one continuous approach that
speeds and slows but never pauses, a terminal creep at 3 mm per step
with the gripper closing *in motion*, and the parking window driven
from v1's 107 steps to a median and maximum of **one**. Camera 3
moved close to the workspace (fixed table, measured pose, the table
and box always framed), the penguin turned blue for contrast, and
scene randomisation was designed so position could never substitute
for reading the instruction. The campaign banked 600 episodes in 600
attempts with zero rejections. Training π₀ on this data (30k steps,
extended to 60k when validation kept improving; checkpoint 047500
selected on a pre-registered rule) produced the project's first
complete closed-loop pick-and-place — but only one in sixty, with a
diagnosis worth more than the number: the policy flew well, came
within 30 mm in seven episodes, and converted almost none of them,
and in sixteen episodes it flew a beautiful approach to the *wrong*
object. The failure separated cleanly into a terminal-servo gap and
a language-grounding gap.

## 4. Fine-tuning on the corrective dataset: pairwise separation and
## the yaw-on-approach correction

The E3 collection attacked both gaps with 310 new demonstrations.
For grounding, **paired-command episodes**: the same layout flown
twice with only the instruction changed, so the sentence is the sole
signal separating the two trajectories — pairwise separation for the
linguistic representation. For the terminal gap, a corrective
flavour designed in review: the demonstrator drifts slightly off the
object line the way the policy does, then **yaws on approach to
re-point the gripper at the target** before the standard creep — so
the dataset finally contains the state the policy actually fails in,
with the correction attached, expressed entirely in the
choreography's own vocabulary. Three fine-tuning recipes bracketed
the problem: the full-rate run churned the model (everything worse),
the gentle run preserved it but learned nothing, and the rebalanced
run (half of every batch new behaviour, ten times the exposure)
became the first to genuinely learn the corrective flavour. Its
teacher-forced "old-behaviour drift" failed a strict pre-registered
proxy, but a documented amendment sent it to the closed-loop gate,
which vindicated it: median miss improved from 175.5 mm to
**109.3 mm** on the paired mini with navigation at 3/4 and zero
contact violations — a better-flying policy whose approaches
tightened everywhere, though pure-policy conversion remained zero.
The grounding probe told the same story from the language side: the
baseline policy heads for the correct object from a standing start
only 19% of the time and almost never changes course when the
command is swapped (3% flip rate); even a churned pass over the
paired data doubled correct-heading and began creating instruction
sensitivity.

## 5. Where we are: the hybrid policy

The consistent finding across every stage is a division of labour.
The learned policy is good at what only a policy can do — reading
the scene, choosing the object, flying the approach — and the final
few centimetres are better served by tight closed-loop control than
by imitation. The hybrid rung embraces that: the policy flies, and
when it brings the drone and gripper close to the target (within
0.15 m), the platform's scripted terminal servo — the same creep law
that banked 910 demonstrations without a rejection — **fires and
completes the grasp**, then hands control back for the carry and
place. It was validated by expert replay (including rescuing a
deliberately stalled pilot) before its first flight, and that first
flight delivered the project's largest single jump: **four complete
pick-and-places in ten episodes (40%)**, against 1.7% for the best
pure policy, with every success placed in the box and both misses
traced to a known, fixable cause (the servo does not yet correct
heading, which penguin grasps require). Two extensions are already
in motion: the same servo on the better-flying fine-tuned policy —
whose approaches enter the trigger radius in eight of ten episodes —
and a *learned* terminal controller trained overnight by
reinforcement against the dense weld reward, so the ladder can
compare a scripted closer and a learned one on equal terms.

## The through-line

Every stage repeats one lesson in a new costume: the system learns
exactly what it is shown, and it is evaluated honestly only by what
it does closed-loop. Overhead reaches taught visibility; v1 taught
that demonstrations transmit habits, not intentions; v2 taught that
smooth data makes smooth policies but does not conjure precision;
the fine-tune taught that new behaviour needs exposure, not just
presence, in the batch; and the hybrid shows that knowing precisely
*where* the learned system's competence ends is itself the
engineering result that unlocks task success.
