"""Record, reproducibly, that iterative difference-of-means deflation cannot
produce more than one direction — which is why the rank sweep was invalid.

    python deflation_degeneracy_check.py --npz decision_activations.npz \
        --meta episodes_meta.json --out deflation_degeneracy.json

WHY THIS FILE EXISTS
--------------------
ablate_sweeps.deflated_directions claims to return n_dirs orthonormal
directions by "iterative difference-of-means with deflation": take the
difference of class means, remove it from the data, take the difference of
means again. The rank sweep built on it returned IDENTICAL results for ranks
1, 2, 4, 8 and 16 — flip +0.1358 and collateral 0.5427 at every rank — which
is the signature of a rank parameter that is not doing anything.

The cause is a mathematical degeneracy rather than a coding slip. Write the
class means as mu_pos and mu_neg and the first direction as

    v = (mu_pos - mu_neg) / ||mu_pos - mu_neg||

Deflation then replaces every sample x by x - (x . v) v. Means are linear, so
the deflated class means are mu_pos - (mu_pos . v) v and mu_neg - (mu_neg . v) v,
and their difference is

    (mu_pos - mu_neg) - ((mu_pos - mu_neg) . v) v

But (mu_pos - mu_neg) is exactly parallel to v by construction, so that
expression is identically zero. The second iteration therefore finds a zero
difference, hits the `if nrm < 1e-9: break` guard, and every direction after
the first is left as zeros. Projecting out a zero vector is a no-op, so a
rank-16 ablation removes exactly the same one-dimensional subspace a rank-1
ablation does.

This is not a quirk of this codebase — it is why the concept-erasure
literature uses methods built for the multi-dimensional case. Iterative
Nullspace Projection refits a fresh CLASSIFIER on the deflated data at each
iteration, which does find new directions because a linear classifier is not
constrained to the mean difference; LEACE solves the whole erasure problem in
closed form with a guarantee.

WHAT IT REPORTS
  n_nonzero_per_layer   how many of the requested directions came back non-zero
  pairwise_cosines      of the returned directions, where more than one exists
  post_deflation_gap    ||mean difference|| after removing direction 0, which
                        the argument above predicts is ~0
  verdict               degenerate / not degenerate

The check is run on the real extracted activations rather than on synthetic
data, so it documents the behaviour of the exact call the rank sweep made.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
from ablate_sweeps import deflated_directions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stack", default="expert", choices=["trunk", "expert"])
    ap.add_argument("--n_dirs", type=int, default=8)
    ap.add_argument("--out", default="deflation_degeneracy.json")
    args = ap.parse_args()

    key = f"{args.stack}_resid_mean"
    pos = lambda r: r["mode"] == "refuse_hazard"      # noqa: E731
    neg = lambda r: r["mode"] == "act"                # noqa: E731
    vecs = deflated_directions(args.npz, args.meta, key, pos, neg, args.n_dirs)
    n_layers = vecs.shape[0]
    norms = np.linalg.norm(vecs, axis=2)              # (layers, n_dirs)
    n_nonzero = (norms > 1e-9).sum(axis=1)

    # Reproduce the argument numerically on the real data: after removing
    # direction 0, is the class-mean difference actually zero?
    z = np.load(args.npz)
    payload = json.loads(pathlib.Path(args.meta).read_text())
    eps = payload["episodes"] if isinstance(payload, dict) else payload
    by = {int(e["episode_index"]): e for e in eps}
    rows = [by[int(i)] for i in z["episode_index"].astype(int)]
    p = np.array([pos(r) for r in rows])
    n = np.array([neg(r) for r in rows])
    arr = z[key].astype(np.float64)

    gaps = []
    for layer in range(n_layers):
        x = arr[:, layer].copy()
        d0 = x[p].mean(0) - x[n].mean(0)
        before = float(np.linalg.norm(d0))
        v = d0 / (before + 1e-30)
        x = x - np.outer(x @ v, v)                    # the deflation step
        after = float(np.linalg.norm(x[p].mean(0) - x[n].mean(0)))
        gaps.append({"layer": layer, "gap_before": before, "gap_after": after,
                     "ratio": after / (before + 1e-30)})

    degenerate = bool(n_nonzero.max() <= 1)
    out = {
        "stack": args.stack,
        "n_dirs_requested": args.n_dirs,
        "n_layers": int(n_layers),
        "n_nonzero_per_layer": n_nonzero.tolist(),
        "n_nonzero_unique": sorted(set(int(v) for v in n_nonzero)),
        "post_deflation_gap": gaps,
        "max_gap_ratio_after_deflation": max(g["ratio"] for g in gaps),
        "degenerate": degenerate,
        "verdict": (
            "DEGENERATE: deflated_directions returns exactly one non-zero "
            "direction regardless of n_dirs, so every rank>1 ablation removed "
            "the same 1-D subspace as rank 1. Multi-dimensional erasure needs "
            "INLP (refit a classifier per iteration) or LEACE (closed form)."
            if degenerate else
            "NOT degenerate: more than one non-zero direction was returned."),
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"requested {args.n_dirs} directions on the {args.stack} stack")
    print(f"  non-zero returned per layer: {sorted(set(int(v) for v in n_nonzero))}")
    print(f"  max ||mean diff|| ratio after removing direction 0: "
          f"{out['max_gap_ratio_after_deflation']:.3e}")
    print(f"  {out['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
