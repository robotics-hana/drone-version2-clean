# Interpretability experiments: which file, and why

Read `V4STYLE_README.md` first — it covers the dataset these consume.

## The claim, and what currently supports it

> Ablating the act-vs-refuse direction in the trained expert degrades attitude
> control (median peak tilt 4.4° → 15.8°), while a norm-matched random
> direction does nothing (p = 0.59).

That is a real causal result about the component that produces actions, and it
stands. Two things it does **not** yet support:

1. **"Refusal is inherently non-separable from control in a VLA."** With a
   frozen trunk, only the expert learned the task, so "the causal direction
   lives in the expert" is true *by construction*, not by measurement.
2. **"The effect is about refusal."** A norm-matched *random* direction only
   rules out "any perturbation of this magnitude breaks flight". It does not
   rule out the live alternative: *ablating any direction that encodes the
   upcoming trajectory degrades control, and refusal is not special.*

Point 2 is the one that decides whether the headline survives review, and it is
the cheapest to close. That is why it is experiment 1 below.

## Files

| file | experiment | why it exists |
| --- | --- | --- |
| `probe_extract_pi0.py` | **0. Activations.** Residual stream at the decision frame → `.npz`. | π0 replacement for `probe_extract.py`, which is SmolVLA-specific in both its layer paths and its residual arithmetic. |
| `direction_controls.py` | **1. Behaviour-matched controls.** Builds `v_refusal` plus `v_band`, `v_relation`, `v_ungrounded`, `v_random`. | A random control is a weak baseline. `v_band` and `v_relation` separate populations that fly *different trajectories* but make the *same decision* — the only controls that can distinguish "refusal is entangled with control" from "trajectory encoding is entangled with control". |
| `ablate_sweeps.py --sweep layer` | **2. Where is the decision computed?** Single-layer ablation across the stack, measuring flip vs collateral. | The existing harness intervenes at L\* *and above*, which answers "does it matter anywhere" but not "where". If no layer flips the decision without collateral damage, non-separability is a property of the whole stack rather than of one hook point — a much stronger claim than a single-site result. |
| `ablate_sweeps.py --sweep rank` | **3. How distributed is it?** Rank-k ablation curve via deflated difference-of-means. | "Decodable but causally inert" (trunk AUROC 0.81, no effect) is most likely *redundant encoding*: a rank-1 projection cannot remove information spread over many directions. "Behaviour flips at rank 8" is a measurement; "it is inert" invites the objection that you looked in the wrong place. |
| `install_sidecar.py` | plumbing | Puts the collector's per-episode metadata where the analysis tools look for it (`<dataset root>/episodes_meta.json`). |

Experiments 2 and 3 share one file because they share the intervention hook.
Two files would mean two copies of the projection code, which is how two copies
drift apart.

**This pipeline is π0-only.** The SmolVLA-era files (`probe_extract.py`,
`ablate_refusal.py`) are legacy: they are not part of this pipeline and cannot
run against π0 without the port applied here. They are left on disk as the
record of how the published numbers were produced, not as a comparison arm.
Where SmolVLA is mentioned below it is only to explain why a π0 file had to
differ from its predecessor.

## π0 accepts the dataset as collected — verified

`PI0Config` ships with empty `input_features`, so features are inferred from
the dataset. Against a real sample from the collected data:

    IN   observation.images.camera1/2/3   VISUAL (3, 480, 640)
    IN   observation.state                STATE  (16,)
    OUT  action                           ACTION (6,)
    cameras accepted 3 · max_state_dim 32 · max_action_dim 32 · chunk 50
    predict_action_chunk -> (1, 50, 6)

So the `camera1/2/3` naming and the 16-D state — both chosen for SmolVLA — carry
over unchanged, and the 6-D action is padded to π0's 32-wide action space. **No
re-collect is needed.** Worth having checked before the run finished, since the
alternative fix would have been re-collecting 730 episodes.

## Order to run

```bash
# 0. after collection: put the metadata where the tools expect it
python install_sidecar.py pick_hold_v4s_train_meta.json hanapasta/pick_hold_v4s_train

# 1. residual stream at the decision frame (GPU, pi0-native)
python probe_extract_pi0.py --checkpoint CKPT \
       --dataset hanapasta/pick_hold_v4s_train \
       --meta pick_hold_v4s_train_meta.json --out decision_activations.npz

# 2. directions + controls (CPU, seconds)
python direction_controls.py --npz decision_activations.npz \
       --meta pick_hold_v4s_train_meta.json --out directions.npz

# 3. the experiment that matters most: does a matched control degrade too?
python ablate_sweeps.py --sweep layer --direction refusal  ... --out sweep_refusal.json
python ablate_sweeps.py --sweep layer --direction band     ... --out sweep_band.json
python ablate_sweeps.py --sweep layer --direction relation ... --out sweep_relation.json

# 4. how distributed
python ablate_sweeps.py --sweep rank --stack trunk  --ranks 1,2,4,8,16 ...
python ablate_sweeps.py --sweep rank --stack expert --ranks 1,2,4,8,16 ...
```

## How to read the results

`direction_controls.py` prints `cosine(refusal, control)` per layer. **If a
control is nearly parallel to `v_refusal`, it is not a control** — it is the
same direction under another name, and ablating it proves nothing. Near-zero
cosine is what makes the comparison meaningful.

In the sweeps, a clean separable refusal feature gives **high flip, ~zero
collateral**. The outcomes and what each means:

| result | reading |
| --- | --- |
| refusal flips, controls do not | the effect is about the decision — headline claim strengthens |
| refusal and `v_band` both degrade | the effect is about trajectory encoding in general — headline is an overclaim and must be rewritten |
| no layer flips without collateral | non-separability holds across the whole stack — the strongest version of the claim |
| trunk needs rank ≫ 1 to flip | the trunk encodes the decision redundantly; "causally inert" was a rank-1 artifact |

## Validation status, and the one thing still broken

**These are unvalidated end to end.** The projection arithmetic is self-tested
(`python ablate_sweeps.py --selftest`, max residual 1.9e-15, rank-1 identical to
the scalar form used by `ablate_refusal.py`), and `direction_controls.py` was
verified on synthetic activations with a planted refusal axis and an orthogonal
planted band axis — it recovered them at cosine 0.986 and 0.938 and reported
them near-orthogonal (0.014). But nothing has been run against a real
checkpoint, because no policy has been trained on this dataset yet.

**Ported to π0 and verified.** `probe_extract_pi0.py` and `ablate_sweeps.py`
are now π0-native. The port was not just a path swap — π0's block ends with a
**gated** residual, `after_first_residual + mlp_out * gate`, not the plain add
`probe_extract.py` assumes. Rewriting the mlp output (the SmolVLA recipe) is
therefore mis-scaled by the gate whenever adarms is on (π0.5), and it fails
*silently*. Both files instead read and write the residual stream directly, at
the input to each block's `input_layernorm`, which is correct either way.

Verified against `lerobot/pi0_base`: 18 layers per stack (trunk width 2048,
expert 1024), pre-hooks fire on 36/36, the L0 capture equals the fed embeddings
exactly, an identity rewrite is bit-exact, and a rank-1 projection moves the
output (relative 0.034). The gate is `None` for `pi0_base`, so the plain-add
reconstruction would coincidentally have worked there — but not for π0.5.

Note layer indices shift by one relative to `probe_extract.py`:
`resid_pre[L] == resid_post[L-1]`, and there is one extra entry (the input to
the final norm).

**The HF token on the cluster has expired** — every `huggingface_hub` call 401s,
including public repos. Cached models load with `HF_HUB_OFFLINE=1`, which is how
the verification above ran, but `push_to_hub` and any new download will fail
until it is refreshed.

## On the frozen trunk

Keep it frozen as the primary configuration: at 480 training episodes a full
trunk finetune will overfit, and freezing keeps the three replicates differing
only in the expert, which is what makes the replication claim clean.

The frozen trunk does **not** pre-install a refusal direction. A safety-tuned
VLM refuses *harmful requests*; "pick up the green block" is not one, and the
placard will not engage it. What the trunk probe reads at AUROC 0.81 is
"is the named object placarded" — the *evidence*, not the decision. Present
that number as an **input-observability check** (alongside the 200–454 px
placard measurement at the decision frame), not as a claim about policy
representations.

To close the "frozen by construction" objection, run **one** LoRA-trunk
replicate (`armc_lora.py` and `armc_merge.py` already exist on the cluster) and
repeat experiments 1–3 on it. If the trunk direction stays causally inert with
an adapted trunk, the non-separability claim is no longer about your training
setup. If it becomes potent, that is a finding too: where refusal lives depends
on what you train.
