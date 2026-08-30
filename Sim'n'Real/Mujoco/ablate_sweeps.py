"""Layer sweep and rank-k sweep for the refusal-direction intervention.

    # where is the decision computed?
    python ablate_sweeps.py --sweep layer --checkpoint CKPT --dataset REPO \
        --npz decision_activations.npz --meta pick_hold_v4s_train_meta.json

    # how distributed is it?
    python ablate_sweeps.py --sweep rank  --checkpoint CKPT --dataset REPO \
        --npz decision_activations.npz --meta ... --ranks 1,2,4,8,16

    # is the effect specific to refusal, or to any trajectory-encoding axis?
    python ablate_sweeps.py --sweep layer --direction band  ...

TWO EXPERIMENTS, ONE FILE, because they share the intervention hook. Splitting
them would mean two copies of the projection code, which is how two copies
drift apart.

WHY -- LAYER SWEEP
ablate_refusal.py intervenes at L* and ABOVE, which answers "does the direction
matter anywhere" but not "where is the decision computed". A single-layer sweep
asks the mechanistic question directly, and it is the experiment that decides
whether refusal is SEPARABLE:

    if some layer flips refuse -> comply while leaving act episodes' actions
    intact, refusal is separable there and you have localised it;
    if no layer anywhere gives a clean flip without collateral damage,
    non-separability is a property of the whole stack rather than an artifact
    of the single hook point that was tried.

The second reading is a far stronger claim than a single-site result, which is
why this sweep is worth running even though the headline number already exists.

WHY -- RANK-K SWEEP
"decodable but causally inert" (trunk probe AUROC 0.81, no behavioural effect)
is almost certainly REDUNDANT ENCODING rather than absence: a rank-1 projection
cannot remove information that is spread over many directions. The rank-k curve
turns an assertion into a measurement -- "the behaviour flips at rank 8" is a
quantitative statement about how distributed the encoding is, whereas "it is
inert" invites the objection that you looked in the wrong place. Directions are
built by iterative difference-of-means with deflation, so each successive
direction captures the separation left after the previous ones are removed.

WHAT IS MEASURED (open loop, at the decision frame)
  flip   : how far the emitted action chunk on REFUSE episodes moves toward the
           act distribution. This is the intended effect.
  collat : how far the emitted action chunk on ACT episodes moves from its own
           baseline. This is collateral damage -- a clean, separable refusal
           feature should score high flip and ~zero collat.
Attitude degradation (the entanglement signature) is only observable in CLOSED
loop, flying the policy in MuJoCo; that is the follow-up, and it is why this
file reports collat as the open-loop proxy rather than claiming to measure
control degradation.

WHERE THE INTERVENTION IS APPLIED
The residual stream entering each block, i.e. the input to
`layer.input_layernorm`, rewritten through a forward-pre hook. NOT the mlp
output, which is what the SmolVLA harness (ablate_refusal.py) rewrites: pi0's
block ends with a GATED residual, `afr + mlp_out * gate`, so rewriting the mlp
output is mis-scaled by the gate whenever adarms is on (pi05). Reading and
writing the stream directly is correct for both.

STATUS: pi0-native. The hook point is VERIFIED against lerobot/pi0_base --
pre-hooks fire on 18/18 layers of both stacks, an identity rewrite is
bit-exact, and a rank-1 projection moves the output. The projection arithmetic
is self-tested (`--selftest`). NOT yet run end to end: no pi0 policy has been
trained on this dataset yet.
"""
import argparse
import json
import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# direction construction (pure numpy -- testable without a policy)

def deflated_directions(npz_path, meta_path, stack_key, pos_sel, neg_sel,
                        n_dirs):
    """n_dirs orthonormal directions by iterative difference-of-means with
    deflation: take the difference of class means, remove it from the data,
    take the difference of means again. Each direction therefore captures the
    class separation that survives removing all previous ones -- which is
    exactly what a rank-k ablation removes."""
    z = np.load(npz_path)
    meta = json.loads(pathlib.Path(meta_path).read_text())
    eps = meta["episodes"] if isinstance(meta, dict) else meta
    by_idx = {int(e["episode_index"]): e for e in eps}
    rows = [by_idx[int(i)] for i in z["episode_index"].astype(int)]
    pos = np.array([pos_sel(r) for r in rows])
    neg = np.array([neg_sel(r) for r in rows])
    arr = z[stack_key].astype(np.float64)            # (n, NL, width)
    NL, width = arr.shape[1], arr.shape[2]
    V = np.zeros((NL, n_dirs, width), np.float32)
    for L in range(NL):
        X = arr[:, L].copy()
        for k in range(n_dirs):
            d = X[pos].mean(0) - X[neg].mean(0)
            nrm = np.linalg.norm(d)
            if nrm < 1e-9:
                break
            v = d / nrm
            V[L, k] = v.astype(np.float32)
            X = X - np.outer(X @ v, v)               # deflate
    return V


def project_out(r, V):
    """r' = r - V V^T r, for V a (k, width) orthonormal-ish basis."""
    if V.ndim == 1:
        V = V[None, :]
    return r - (r @ V.T) @ V


def selftest():
    """The projection arithmetic, independently of any model."""
    rng = np.random.default_rng(0)
    W, k, n = 64, 3, 50
    V = np.linalg.qr(rng.normal(size=(W, k)))[0].T      # (k, W) orthonormal
    r = rng.normal(size=(n, W))
    out = project_out(r, V)
    resid = np.abs(out @ V.T).max()
    reproj = np.abs(project_out(out, V) - out).max()
    print(f"[selftest] max |V^T r'| after projection : {resid:.2e} (want ~0)")
    print(f"[selftest] idempotent (P^2 = P) max err  : {reproj:.2e} (want ~0)")
    # rank-1 must equal the scalar form used by ablate_refusal
    v = V[0]
    scalar = r - np.outer(r @ v, v)
    print(f"[selftest] rank-1 == scalar form max err : "
          f"{np.abs(scalar - project_out(r, v[None])).max():.2e} (want ~0)")
    assert resid < 1e-10 and reproj < 1e-10
    print("[selftest] OK")


# ---------------------------------------------------------------------------
# intervention (needs the policy)

def run_sweep(args):
    import torch
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = LeRobotDataset(args.dataset)
    meta = json.loads(pathlib.Path(args.meta).read_text())
    eps = meta["episodes"] if isinstance(meta, dict) else meta

    pwe = policy.model.paligemma_with_expert
    stacks = {"trunk": pwe.paligemma.model.language_model.layers,
              "expert": pwe.gemma_expert.model.layers}

    sel = {
        "refusal": (lambda r: r["mode"] == "refuse_hazard",
                    lambda r: r["mode"] == "act"),
        "band": (lambda r: r["mode"] == "act" and r.get("band") == "far",
                 lambda r: r["mode"] == "act" and r.get("band") == "near"),
        "relation": (lambda r: r["mode"] == "act" and r.get("relation") == "R1",
                     lambda r: r["mode"] == "act" and r.get("relation") == "R2"),
    }[args.direction]
    key = f"{args.stack}_resid_mean"
    n_dirs = max([int(x) for x in args.ranks.split(",")]) if args.sweep == "rank" else 1
    V = deflated_directions(args.npz, args.meta, key, *sel, n_dirs)
    print(f"[dirs] {args.direction} on {args.stack}: {V.shape}")

    # --- hooks -------------------------------------------------------------
    # Intervene on the INPUT to each block's input_layernorm. That tensor IS
    # the residual stream entering the block, whatever the block does
    # internally -- which matters for pi0, whose block ends with a GATED
    # residual (afr + mlp_out * gate) rather than a plain add. Rewriting the
    # mlp output, as the SmolVLA harness does, would be mis-scaled by the gate
    # under pi05/adarms and silently wrong. Verified on lerobot/pi0_base:
    # pre-hooks fire 18/18 on both stacks, an identity rewrite is bit-exact,
    # and a rank-1 projection moves the output.
    handles = []

    def rewrite(Vk):
        Vt = torch.tensor(Vk, dtype=torch.float32)

        def h(_m, a):
            x = a[0]
            Vd = Vt.to(x.device, x.dtype)
            return (x - (x @ Vd.T) @ Vd,) + tuple(a[1:])
        return h

    def install(layers_sel, rank):
        for stack, Ls in layers_sel.items():
            for L in Ls:
                handles.append(stacks[stack][L].input_layernorm
                               .register_forward_pre_hook(rewrite(V[L, :rank])))

    def clear():
        while handles:
            handles.pop().remove()

    def emit(rows):
        acts = []
        for e in rows:
            erow = ds.meta.episodes[int(e["episode_index"])]
            g = int(erow["dataset_from_index"]) + int(e["decision_frame"])
            item = ds[g]
            raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                   for k, v in item.items()}
            with torch.no_grad():
                a = policy.predict_action_chunk(preprocessor(raw))
            acts.append(a[0].float().cpu().numpy())
        return np.stack(acts)

    hz = [e for e in eps if e["mode"] == "refuse_hazard"][:args.n_episodes]
    ac = [e for e in eps if e["mode"] == "act"][:args.n_episodes]
    base_hz, base_ac = emit(hz), emit(ac)
    act_centroid = base_ac.mean(0)
    d0 = np.linalg.norm(base_hz - act_centroid, axis=(1, 2)).mean()

    NL = V.shape[0]
    results = []
    sweep = (range(NL) if args.sweep == "layer"
             else [int(x) for x in args.ranks.split(",")])
    for s in sweep:
        if args.sweep == "layer":
            install({args.stack: [s]}, 1)
            label = f"L{s}"
        else:
            install({args.stack: list(range(args.layer_lo, NL))}, s)
            label = f"rank{s}"
        try:
            hz_i, ac_i = emit(hz), emit(ac)
        finally:
            clear()
        # flip: fraction of the baseline refuse-to-act gap that was closed
        d1 = np.linalg.norm(hz_i - act_centroid, axis=(1, 2)).mean()
        flip = float((d0 - d1) / d0) if d0 > 0 else 0.0
        collat = float(np.linalg.norm(ac_i - base_ac, axis=(1, 2)).mean()
                       / (np.linalg.norm(base_ac, axis=(1, 2)).mean() + 1e-9))
        results.append({"step": label, "flip_frac": flip, "collateral": collat})
        print(f"  {label:8s} flip {flip:+.3f}   collateral {collat:.3f}",
              flush=True)

    out = {"sweep": args.sweep, "direction": args.direction,
           "stack": args.stack, "checkpoint": args.checkpoint,
           "n_hazard": len(hz), "n_act": len(ac), "results": results}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=("layer", "rank"), default="layer")
    ap.add_argument("--direction", choices=("refusal", "band", "relation"),
                    default="refusal",
                    help="'band'/'relation' are the behaviour-matched controls")
    ap.add_argument("--stack", choices=("trunk", "expert"), default="expert")
    ap.add_argument("--checkpoint")
    ap.add_argument("--dataset")
    ap.add_argument("--npz")
    ap.add_argument("--meta")
    ap.add_argument("--ranks", default="1,2,4,8,16")
    ap.add_argument("--layer_lo", type=int, default=8)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--out", default="sweep_results.json")
    ap.add_argument("--selftest", action="store_true",
                    help="check the projection arithmetic and exit")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    for r in ("checkpoint", "dataset", "npz", "meta"):
        if not getattr(args, r):
            ap.error(f"--{r} is required (or pass --selftest)")
    run_sweep(args)


if __name__ == "__main__":
    main()
