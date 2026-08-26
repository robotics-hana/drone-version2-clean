# Abstract (draft)

> DRAFT. Bracketed [X] values are placeholders for the Phase E ladder
> results (running after the 30k checkpoint completes) — fill them in
> before submission, and rewrite in your own voice.

Vision-language-action (VLA) models trained on ground-robot data have
recently been transferred to aerial manipulation: Tucker et al. (2026)
report that a fine-tuned π₀ policy, combined with two purely
inference-time mechanisms — Real-Time Chunking (RTC) and a physics-based
Payload-Aware Guidance term injected into the flow-matching sampler —
progresses from 50%/0% pick/place under naive chunk execution to
100%/50% on a real quadrotor. This dissertation asks whether those
gains are properties of the *method* or of the *embodiment*, by
recreating the complete pipeline on a different platform: SkyGrip, a
quadrotor carrying an articulated 2-DoF arm and a 32 mm parallel-jaw
gripper, in a metrically registered simulation of a real laboratory.

The recreation reproduces the paper's recipe wherever it is specified —
dataset composition (120 manipulation, 150 navigation, 50 corrective
episodes), full fine-tuning of the public π₀ base checkpoint under the
published optimisation schedule, and the three-method inference ladder
evaluated on staged success criteria including a held-out compositional
prompt — and records every unstated design decision in a 36-entry
decision log. Demonstrations come from a scripted expert engineered for
physical honesty: every banked grasp passes measurement-based gates that
a real gripper could satisfy, yielding a 320-episode dataset collected
at a 100% bank rate with zero discards.

Across 20 trials per task per method, the ladder [reproduced /
partially reproduced] the paper's ordering: pick/place rose from
[X]%/[X]% (naive) to [X]%/[X]% (+RTC) to [X]%/[X]% (+guidance), with
gate navigation at [X]% and the held-out compositional prompt at [X]%.
Near-miss instrumentation attributes the residual failures to the
platform's grasp-precision demands — a ±5 mm jaw window roughly three
times tighter than the original platform's compliant gripper on a plush
target — [supporting / qualifying] the paper's central claim: the
inference-time ladder transfers across embodiment, while its absolute
ceilings are set by the embodiment's tightest physical tolerance.
