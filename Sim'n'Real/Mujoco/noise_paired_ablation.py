"""Noise-paired directional ablation for pi0, plus the instrument audit that
motivates it.

    python noise_paired_ablation.py --stage audit --out audit.json ...
    python noise_paired_ablation.py --stage sweep --direction band ...

WHY THIS FILE EXISTS (it does NOT replace ablate_sweeps.py)
-----------------------------------------------------------
ablate_sweeps.py produced four layer sweeps on checkpoint 060000 in which the
refusal direction, the band direction and a norm-matched RANDOM direction were
all indistinguishable:

    refusal(expert)  mean flip +0.015   mean collateral 0.618
    band(expert)     mean flip +0.028   mean collateral 0.601
    random(expert)   mean flip +0.031   mean collateral 0.587

The band direction is strongly encoded at this checkpoint (cross-validated
AUROC 0.969 against a 0.512 permutation null), so it was serving as the
POSITIVE CONTROL. A positive control that lands on top of the random arm does
not license the conclusion "refusal is not causally used". It says only that
the measurement could not have detected the effect either way.

The suspected cause is that pi0 is a FLOW-MATCHING policy. Its action chunk is
produced by integrating from a noise sample:

    modeling_pi0.sample_actions()  noise = self.sample_noise(shape, device)
    modeling_pi0.sample_noise()    torch.normal(0., 1., size=shape)  # no generator

There is no generator and no seed, so every call to predict_action_chunk draws
FRESH noise. ablate_sweeps.py compares one clean call against one intervened
call, so its "collateral" statistic

    ||chunk_intervened - chunk_baseline|| / ||chunk_baseline||

contains the sampling variance of the policy PLUS the causal effect of the
ablation, with no way to separate them. If the sampling term dominates, every
arm converges on the same number -- which is exactly the observed pattern.

sample_actions() accepts noise=..., and predict_action_chunk forwards **kwargs
to it, so the noise CAN be pinned. This file does that.

WHAT IT MEASURES
----------------
--stage audit establishes the noise floor before any causal claim is made:

  null_unpaired  two clean passes, fresh noise each time (what ablate_sweeps.py
                 was implicitly comparing against). No hook is installed, so
                 any non-zero value here is pure sampling variance.
  null_paired    two clean passes with the SAME pinned noise. This must come
                 out at ~0. It is the end-to-end determinism check: it proves
                 the dataset decode, the preprocessor and the forward pass are
                 all deterministic once the noise is fixed, so a non-zero
                 number in the sweep is attributable to the intervention and
                 nothing else.

  The ratio null_unpaired / null_paired is the instrument gain that pinning
  the noise buys.

--stage sweep then repeats the layer (or rank) sweep with the noise pinned, so
baseline and intervened chunks differ ONLY by the projection. Metric
definitions are copied verbatim from ablate_sweeps.py so that paired and
unpaired numbers stay directly comparable.

Directions are imported from ablate_sweeps.deflated_directions rather than
reimplemented, so the vectors under test are bit-identical to the ones the
original sweep used. The only behavioural differences are the pinned noise and
the layer-count fix below.

FIX CARRIED OVER
----------------
probe_extract_pi0.py stores NL = n_blocks + 1 entries per stack (the input to
every block, plus the input to the final norm). Only the first n_blocks are
hookable. ablate_sweeps.py guards this in the LAYER path but not in the RANK
path, where it still iterates range(layer_lo, NL) -- which is why
sweep_rank_refusal_expert.json was never written (IndexError: index 18 is out
of range, job 248198). Both paths clamp to N_HOOKABLE here.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

# Reuse the exact direction construction the original sweep used.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
from ablate_sweeps import deflated_directions  # noqa: E402

# Population selectors, copied verbatim from ablate_sweeps.run_sweep so the
# contrasts under test are identical.
SEL = {
    "refusal": (lambda r: r["mode"] == "refuse_hazard",
                lambda r: r["mode"] == "act"),
    "band": (lambda r: r["mode"] == "act" and r.get("band") == "far",
             lambda r: r["mode"] == "act" and r.get("band") == "near"),
    "relation": (lambda r: r["mode"] == "act" and r.get("relation") == "R1",
                 lambda r: r["mode"] == "act" and r.get("relation") == "R2"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "sweep"], required=True)
    ap.add_argument("--sweep", choices=["layer", "rank"], default="layer")
    ap.add_argument("--direction", default="refusal")
    ap.add_argument("--stack", default="expert", choices=["trunk", "expert"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--ranks", default="1,2,4,8,16")
    ap.add_argument("--layer_lo", type=int, default=0)
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--repeats", type=int, default=3,
                    help="audit only: independent unpaired draws, to put an "
                         "error bar on the noise floor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = (LeRobotDataset(args.dataset, root=args.root) if args.root
          else LeRobotDataset(args.dataset))
    meta = json.loads(pathlib.Path(args.meta).read_text())
    eps = meta["episodes"] if isinstance(meta, dict) else meta

    pwe = policy.model.paligemma_with_expert
    stacks = {"trunk": pwe.paligemma.model.language_model.layers,
              "expert": pwe.gemma_expert.model.layers}
    n_hookable = len(stacks[args.stack])

    hz = [e for e in eps if e["mode"] == "refuse_hazard"][:args.n_episodes]
    ac = [e for e in eps if e["mode"] == "act"][:args.n_episodes]

    # One pinned noise tensor per episode. The same episode always gets the
    # same noise, so baseline and intervened runs are paired per episode;
    # different episodes get different noise, so nothing is special about one
    # particular draw.
    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise_map = {}
    for e in hz + ac:
        noise_map[int(e["episode_index"])] = torch.normal(
            0.0, 1.0, size=(1, chunk, adim), generator=gen).to(dev)

    def emit(rows, pinned):
        """Emit one action chunk per episode. pinned=True passes the episode's
        fixed noise into sample_actions; pinned=False lets pi0 draw its own."""
        acts = []
        for e in rows:
            erow = ds.meta.episodes[int(e["episode_index"])]
            gidx = int(erow["dataset_from_index"]) + int(e["decision_frame"])
            item = ds[gidx]
            raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                   for k, v in item.items()}
            kw = {"noise": noise_map[int(e["episode_index"])]} if pinned else {}
            with torch.no_grad():
                a = policy.predict_action_chunk(preprocessor(raw), **kw)
            acts.append(a[0].float().cpu().numpy())
        return np.stack(acts)

    def collateral(new, base):
        """Relative displacement of the act-episode chunks. Identical formula
        to ablate_sweeps.py so paired and unpaired numbers are comparable."""
        return float(np.linalg.norm(new - base, axis=(1, 2)).mean()
                     / (np.linalg.norm(base, axis=(1, 2)).mean() + 1e-9))

    # ---------------- stage: audit -----------------------------------------
    if args.stage == "audit":
        out = {"checkpoint": args.checkpoint, "n_hazard": len(hz),
               "n_act": len(ac), "noise_seed": args.noise_seed}

        # Sampling variance alone: no hook, fresh noise on both passes. This is
        # the floor the original sweep was unknowingly measuring against.
        unp = []
        for _ in range(args.repeats):
            a1, a2 = emit(ac, False), emit(ac, False)
            unp.append(collateral(a2, a1))
            print(f"  null_unpaired {unp[-1]:.4f}", flush=True)
        out["null_unpaired"] = {"values": unp,
                                "mean": float(np.mean(unp)),
                                "std": float(np.std(unp))}

        # Determinism check: no hook, SAME pinned noise on both passes.
        p1, p2 = emit(ac, True), emit(ac, True)
        out["null_paired"] = collateral(p2, p1)
        print(f"  null_paired   {out['null_paired']:.6f}", flush=True)

        # The same two statistics for the decision metric (flip), so the
        # sweep's flip column gets a noise floor of its own.
        b_hz, b_ac = emit(hz, True), emit(ac, True)
        cen = b_ac.mean(0)
        d0 = np.linalg.norm(b_hz - cen, axis=(1, 2)).mean()
        u_hz = emit(hz, False)
        d1u = np.linalg.norm(u_hz - cen, axis=(1, 2)).mean()
        out["flip_null_unpaired"] = float((d0 - d1u) / d0)
        out["baseline_gap_d0"] = float(d0)
        print(f"  flip_null_unpaired {out['flip_null_unpaired']:+.4f}",
              flush=True)

        out["verdict"] = (
            "Pinning the noise removes the sampling term if null_paired is "
            "~0 while null_unpaired is large. The ratio is how much of the "
            "original sweep's collateral was noise.")
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
        return

    # ---------------- stage: sweep -----------------------------------------
    key = f"{args.stack}_resid_mean"
    n_dirs = (max(int(x) for x in args.ranks.split(","))
              if args.sweep == "rank" else 1)
    if args.direction == "random":
        z = np.load(args.npz)
        arr = z[key]
        rng = np.random.default_rng(0)        # same seed as ablate_sweeps.py
        vecs = np.zeros((arr.shape[1], n_dirs, arr.shape[2]), np.float32)
        for layer in range(arr.shape[1]):
            for k in range(n_dirs):
                v = rng.normal(size=arr.shape[2])
                vecs[layer, k] = (v / np.linalg.norm(v)).astype(np.float32)
    else:
        vecs = deflated_directions(args.npz, args.meta, key,
                                   *SEL[args.direction], n_dirs)
    print(f"[dirs] {args.direction} on {args.stack}: {vecs.shape}  "
          f"hookable {n_hookable}", flush=True)

    handles = []

    def rewrite(vk):
        vt = torch.tensor(vk, dtype=torch.float32)

        def hook(_m, a):
            x = a[0]
            vd = vt.to(x.device, x.dtype)
            return (x - (x @ vd.T) @ vd,) + tuple(a[1:])
        return hook

    def install(layers, rank):
        for layer in layers:
            handles.append(stacks[args.stack][layer].input_layernorm
                           .register_forward_pre_hook(rewrite(vecs[layer, :rank])))

    def clear():
        while handles:
            handles.pop().remove()

    # Baseline computed ONCE with pinned noise and reused for every step, so
    # the comparison is exactly paired.
    base_hz, base_ac = emit(hz, True), emit(ac, True)
    cen = base_ac.mean(0)
    d0 = float(np.linalg.norm(base_hz - cen, axis=(1, 2)).mean())

    if args.sweep == "layer":
        steps = [(f"L{L}", [L], 1) for L in range(args.layer_lo, n_hookable)]
    else:
        # RANK path clamped to n_hookable -- this is the IndexError fix.
        steps = [(f"rank{k}", list(range(args.layer_lo, n_hookable)), k)
                 for k in (int(x) for x in args.ranks.split(","))]

    results = []
    for label, layers, rank in steps:
        install(layers, rank)
        try:
            hz_i, ac_i = emit(hz, True), emit(ac, True)
        finally:
            clear()
        d1 = float(np.linalg.norm(hz_i - cen, axis=(1, 2)).mean())
        flip = (d0 - d1) / d0 if d0 > 0 else 0.0
        col = collateral(ac_i, base_ac)
        results.append({"step": label, "flip_frac": flip, "collateral": col})
        print(f"  {label:8s} flip {flip:+.4f}   collateral {col:.4f}",
              flush=True)

    out = {"sweep": args.sweep, "direction": args.direction,
           "stack": args.stack, "checkpoint": args.checkpoint,
           "paired_noise": True, "noise_seed": args.noise_seed,
           "n_hazard": len(hz), "n_act": len(ac),
           "baseline_gap_d0": d0, "results": results}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
