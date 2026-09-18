"""Causal SUFFICIENCY test: add a direction to act episodes and see whether the
policy starts abstaining. Noise-paired throughout.

    python steer_direction.py --direction ungrounded --alphas 0.5,1,2,4 ...

WHY THIS FILE EXISTS
--------------------
Ablation asks whether a direction is NECESSARY: remove it, does behaviour
change. Steering asks whether it is SUFFICIENT: add it, does the behaviour
appear. They fail in different ways, and on this checkpoint the ablation arm
is the weaker of the two, because the effect it is looking for is small enough
to sit inside the noise floor of a 10-step flow-matching rollout.

The dissociation this is built to test is specific. On the expert stack:

  refuse_ungrounded vs act   CV AUROC 0.684 (null 0.474)   encoded
  refuse_hazard     vs act   CV AUROC 0.551 (null 0.496)   at chance
  cosine between the two directions               +0.934   ONE shared axis

and, from channel_dissociation.py, the two refusal types have IDENTICAL
demonstration targets -- joint2 delta -0.8362 for both, because both fly the
same abstain trajectory by design. So the difference between them is not what
the policy was asked to output. It is only how hard the TRIGGER is to compute:
"named colour absent from the scene" versus "the placard belongs to the named
object rather than to a distractor".

That yields a sharp, falsifiable prediction. If the refusal axis is present
and functional and the hazard cue merely fails to drive it, then injecting
that axis into ACT episodes should push the policy toward the abstain
trajectory -- most visibly on joint2, the channel that carries the decision.

  induced joint2 shift toward -0.836   -> the axis is causally sufficient, and
      the hazard failure is a failure to ACTIVATE it, not an absent
      representation
  no shift at any alpha, while a random direction of the same norm does the
      same nothing -> the axis is not a control signal at all and the
      dissociation is descriptive only

The random arm is what makes either reading possible: it says how much of any
observed shift is just what perturbing the residual stream by that magnitude
does. The band arm is a second reference -- band is the most strongly encoded
variable at this checkpoint (0.969), so it shows what a genuinely load-bearing
direction looks like under the same treatment.

ALPHA IS SCALED PER LAYER. Residual norms differ by an order of magnitude
across layers and by ~50x between the trunk and the expert, so a raw alpha is
not comparable anywhere. alpha is expressed in units of the mean residual norm
AT THAT LAYER, measured from the extraction npz, so alpha=1 means "add a
vector as long as a typical residual".

METRICS
  induce_frac    fraction of the act->refuse gap closed, mirroring the flip
                 statistic in the ablation sweeps so the two are comparable
  joint2_shift   mean joint2 of the steered act chunks minus the unsteered
                 baseline, in real units, against the -0.836 the
                 demonstrations ask for. This is the interpretable one.
  collateral     relative displacement of the act chunks, to show the
                 perturbation size that bought the shift
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
from ablate_sweeps import deflated_directions  # noqa: E402

# joint2 is channel 4 of [x, y, z, joint1, joint2, gripper]. It is the channel
# the demonstrations use to express the decision.
JOINT2 = 4

SEL = {
    "refusal": (lambda r: r["mode"] == "refuse_hazard",
                lambda r: r["mode"] == "act"),
    "ungrounded": (lambda r: r["mode"] == "refuse_ungrounded",
                   lambda r: r["mode"] == "act"),
    "any_refusal": (lambda r: r["mode"] != "act",
                    lambda r: r["mode"] == "act"),
    "band": (lambda r: r["mode"] == "act" and r.get("band") == "far",
             lambda r: r["mode"] == "act" and r.get("band") == "near"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="ungrounded",
                    choices=list(SEL) + ["random"])
    ap.add_argument("--stack", default="expert", choices=["trunk", "expert"])
    ap.add_argument("--layers", default="all",
                    help="'all', or comma-separated layer indices")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--refuse_mode", default="refuse_hazard",
                    help="which refusal population defines the target centroid")
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = (LeRobotDataset(args.dataset, root=args.root) if args.root
          else LeRobotDataset(args.dataset))
    payload = json.loads(pathlib.Path(args.meta).read_text())
    eps = payload["episodes"] if isinstance(payload, dict) else payload

    pwe = policy.model.paligemma_with_expert
    stacks = {"trunk": pwe.paligemma.model.language_model.layers,
              "expert": pwe.gemma_expert.model.layers}
    n_hookable = len(stacks[args.stack])

    key = f"{args.stack}_resid_mean"
    z = np.load(args.npz)
    if args.direction == "random":
        rng = np.random.default_rng(0)
        arr = z[key]
        vecs = np.zeros((arr.shape[1], 1, arr.shape[2]), np.float32)
        for layer in range(arr.shape[1]):
            v = rng.normal(size=arr.shape[2])
            vecs[layer, 0] = (v / np.linalg.norm(v)).astype(np.float32)
    else:
        vecs = deflated_directions(args.npz, args.meta, key, *SEL[args.direction], 1)

    # Per-layer residual norm, so alpha is comparable across layers/stacks.
    norms = np.linalg.norm(z[key], axis=2).mean(axis=0)      # (NL+1,)
    layers = (list(range(n_hookable)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    layers = [L for L in layers if L < n_hookable]
    print(f"[dirs] {args.direction} on {args.stack}: {vecs.shape}; "
          f"steering layers {layers[0]}..{layers[-1]}; "
          f"mean |resid| {norms[layers].mean():.2f}", flush=True)

    hz = [e for e in eps if e["mode"] == args.refuse_mode
          and e.get("decision_frame", -1) >= 0][:args.n_episodes]
    ac = [e for e in eps if e["mode"] == "act"
          and e.get("decision_frame", -1) >= 0][:args.n_episodes]

    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise_map = {int(e["episode_index"]): torch.normal(
        0.0, 1.0, size=(1, chunk, adim), generator=gen).to(dev) for e in hz + ac}

    handles = []

    def adder(vec, scale):
        vt = torch.tensor(vec * scale, dtype=torch.float32)

        def hook(_m, a):
            x = a[0]
            return (x + vt.to(x.device, x.dtype),) + tuple(a[1:])
        return hook

    def install(alpha):
        for layer in layers:
            handles.append(stacks[args.stack][layer].input_layernorm
                           .register_forward_pre_hook(
                               adder(vecs[layer, 0], alpha * norms[layer])))

    def clear():
        while handles:
            handles.pop().remove()

    def emit(rows):
        acts = []
        for e in rows:
            erow = ds.meta.episodes[int(e["episode_index"])]
            gidx = int(erow["dataset_from_index"]) + int(e["decision_frame"])
            item = ds[gidx]
            raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                   for k, v in item.items()}
            with torch.no_grad():
                a = policy.predict_action_chunk(
                    preprocessor(raw), noise=noise_map[int(e["episode_index"])])
                a = postprocessor(a)      # real units, so joint2 is comparable
            acts.append(a[0].float().cpu().numpy())
        return np.stack(acts)

    base_ac, base_hz = emit(ac), emit(hz)
    refuse_centroid = base_hz.mean(0)
    d0 = float(np.linalg.norm(base_ac - refuse_centroid, axis=(1, 2)).mean())
    j2_act = float(base_ac[..., JOINT2].mean())
    j2_ref = float(base_hz[..., JOINT2].mean())
    print(f"[base] gap to refuse centroid {d0:.4f} | "
          f"joint2 act {j2_act:+.4f} refuse {j2_ref:+.4f} "
          f"(target shift {j2_ref - j2_act:+.4f})", flush=True)

    results = []
    for alpha in (float(x) for x in args.alphas.split(",")):
        install(alpha)
        try:
            steered = emit(ac)
        finally:
            clear()
        d1 = float(np.linalg.norm(steered - refuse_centroid, axis=(1, 2)).mean())
        induce = (d0 - d1) / d0 if d0 > 0 else 0.0
        j2 = float(steered[..., JOINT2].mean())
        col = float(np.linalg.norm(steered - base_ac, axis=(1, 2)).mean()
                    / (np.linalg.norm(base_ac, axis=(1, 2)).mean() + 1e-9))
        results.append({"alpha": alpha, "induce_frac": induce,
                        "joint2_steered": j2, "joint2_shift": j2 - j2_act,
                        "collateral": col})
        print(f"  alpha {alpha:5.2f}  induce {induce:+.4f}  "
              f"joint2 {j2:+.4f} (shift {j2 - j2_act:+.4f})  "
              f"collateral {col:.4f}", flush=True)

    out = {"direction": args.direction, "stack": args.stack,
           "layers": layers, "refuse_mode": args.refuse_mode,
           "checkpoint": args.checkpoint, "paired_noise": True,
           "n_act": len(ac), "n_refuse": len(hz),
           "baseline_gap": d0, "joint2_act": j2_act, "joint2_refuse": j2_ref,
           "joint2_target_shift": j2_ref - j2_act, "results": results}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
