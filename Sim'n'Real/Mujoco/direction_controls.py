"""Build the refusal direction AND the behaviour-matched control directions.

    python direction_controls.py --npz decision_activations.npz \
                                 --meta pick_hold_v4s_train_meta.json \
                                 --out directions.npz

WHY THIS EXISTS
---------------
The published ablation result is: projecting the act-vs-refuse direction out of
the expert's residual stream degrades attitude control, while a norm-matched
RANDOM direction of equal magnitude does nothing.

A random direction is a weak baseline. It rules out "any perturbation of this
size breaks flight" and nothing else. It does NOT rule out the alternative that
actually threatens the claim:

    ablating ANY direction that encodes the upcoming trajectory degrades
    control, and refusal is not special.

That alternative is only excluded by a control direction which separates two
populations that fly DIFFERENT TRAJECTORIES but make the SAME DECISION. Then:

    refusal ablation degrades control, matched control does not   -> the effect
        is about the decision, and the entanglement claim stands
    both degrade control                                          -> the effect
        is about trajectory encoding in general, and the headline claim is an
        overclaim that has to be rewritten

The dataset is built to supply exactly these populations:

  v_band      far-band ACT  minus  near-band ACT
              Both are acts, both pick up the named object, both end holding it
              aloft at the centre hover. They differ in the LATERAL TRAVERSE
              they must fly -- ~0.38 m vs ~0.19 m of lateral offset between
              target and distractor. Different motion, identical decision.

  v_relation  R1 ACT  minus  R2 ACT
              R1 = distractor shares the SHAPE (colour must discriminate);
              R2 = distractor shares the COLOUR (shape must discriminate).
              Same decision, same trajectory family, different visual binding
              problem. This is the control for "is the direction just encoding
              which attribute the instruction leaned on?"

  v_random    norm-matched random (the original, weak control -- kept so the
              new results are comparable to the published ones)

All four are returned per layer per stack, unit-normalised, in exactly the
format ablate_refusal.build_directions produces, so they drop straight into the
existing intervention harness.

INPUTS
  --npz   decision_activations.npz from probe_extract.py: per-episode residual
          stream at the DECISION FRAME, shape (n, n_layers, width) per stack.
  --meta  the collector's sidecar (or the dataset's episodes_meta.json). Joined
          on episode_index, because the npz carries mode but not band/relation.

This script is pure numpy -- no GPU, no policy load -- so it can be run and
checked long before any intervention is attempted.
"""
import argparse
import json
import pathlib

import numpy as np

STACKS = (("trunk", "trunk_resid_mean"), ("expert", "expert_resid_mean"))


def _unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def _auroc(pos, neg):
    """Rank-based AUROC of a 1-D score. Sanity check that a direction actually
    separates the populations it was built from."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_meta(path):
    d = json.loads(pathlib.Path(path).read_text())
    eps = d["episodes"] if isinstance(d, dict) else d
    return {int(e["episode_index"]): e for e in eps}


def build(npz_path, meta_path, layer_lo=0, seed=0):
    z = np.load(npz_path)
    meta = load_meta(meta_path)
    ep_idx = z["episode_index"].astype(int)
    rows = [meta.get(int(i)) for i in ep_idx]
    missing = sum(r is None for r in rows)
    if missing:
        raise SystemExit(f"{missing}/{len(rows)} npz episodes absent from "
                         f"{meta_path} -- wrong sidecar for this npz?")

    mode = np.array([r["mode"] for r in rows])
    band = np.array([r.get("band", "") for r in rows])
    rel = np.array([r.get("relation", "") for r in rows])
    is_act = mode == "act"

    masks = {
        # headline: the decision contrast
        "refusal": (mode == "refuse_hazard", is_act),
        # behaviour-matched controls: same decision, different behaviour
        "band": (is_act & (band == "far"), is_act & (band == "near")),
        "relation": (is_act & (rel == "R1"), is_act & (rel == "R2")),
        # second decision contrast, different REASON for refusing -- tells you
        # whether the direction is "refuse" or specifically "placard seen"
        "ungrounded": (mode == "refuse_ungrounded", is_act),
    }
    counts = {k: (int(p.sum()), int(n.sum())) for k, (p, n) in masks.items()}
    print("populations (positive vs negative):")
    for k, (a, b) in counts.items():
        print(f"  {k:11s} {a:4d} vs {b:4d}")
    for k, (a, b) in counts.items():
        if min(a, b) < 8:
            print(f"  WARNING: '{k}' has only {min(a,b)} episodes on one side; "
                  f"its direction will be noise-dominated")

    rng = np.random.default_rng(seed)
    out, report = {}, {}
    for stack, key in STACKS:
        if key not in z:
            print(f"  (npz has no '{key}', skipping {stack})")
            continue
        arr = z[key].astype(np.float64)          # (n, NL, width)
        NL, width = arr.shape[1], arr.shape[2]
        for name, (pos, neg) in masks.items():
            if min(pos.sum(), neg.sum()) == 0:
                continue
            D = np.zeros((NL, width), np.float32)
            aur = np.full(NL, np.nan)
            for L in range(NL):
                if L < layer_lo:
                    continue
                d = arr[pos, L].mean(0) - arr[neg, L].mean(0)
                D[L] = _unit(d)
                proj = arr[:, L] @ D[L]
                aur[L] = _auroc(proj[pos], proj[neg])
            out[f"{stack}.{name}"] = D
            report[f"{stack}.{name}_auroc"] = aur
        # norm-matched random, and the per-layer residual norm the add-condition
        # needs (trunk residuals are ~50x the expert's, so a raw alpha is not
        # comparable across stacks)
        R = np.stack([_unit(rng.normal(size=width)) for _ in range(NL)])
        out[f"{stack}.random"] = R.astype(np.float32)
        out[f"{stack}.norms"] = np.array(
            [float(np.linalg.norm(arr[:, L], axis=1).mean()) for L in range(NL)],
            np.float32)

    # How aligned is each control with the refusal direction? If a control is
    # nearly parallel to v_refusal it is not a control at all -- it is the same
    # direction under another name, and ablating it proves nothing.
    print("\ncosine(refusal, control) per layer -- a good control is near 0:")
    for stack, _ in STACKS:
        if f"{stack}.refusal" not in out:
            continue
        vr = out[f"{stack}.refusal"]
        for name in ("band", "relation", "ungrounded", "random"):
            k = f"{stack}.{name}"
            if k not in out:
                continue
            cos = np.einsum("lw,lw->l", vr, out[k])
            good = np.isfinite(cos)
            print(f"  {stack:6s} {name:11s} mean {np.nanmean(cos[good]):+.3f}  "
                  f"max |cos| {np.nanmax(np.abs(cos[good])):.3f}")

    print("\nAUROC of each direction on its own contrast (>0.5 = separates):")
    for k, v in report.items():
        finite = v[np.isfinite(v)]
        if len(finite):
            print(f"  {k:28s} best {finite.max():.3f} @L{int(np.nanargmax(v))}"
                  f"  mean {finite.mean():.3f}")
    out.update(report)
    out["counts"] = json.dumps(counts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="decision_activations.npz from probe_extract.py")
    ap.add_argument("--meta", required=True,
                    help="collector sidecar, joined on episode_index")
    ap.add_argument("--layer_lo", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="directions.npz")
    args = ap.parse_args()
    out = build(args.npz, args.meta, args.layer_lo, args.seed)
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
