"""Selection-aware cross-validated probe: corrects the layer-selection bias in
probe_cv.py, and can score the emitted action chunk with the same machinery.

    python probe_cv_strict.py --npz decision_activations_pinned.npz \
        --meta episodes_meta.json --perms 100 --out probe_cv_strict.json

WHY THIS FILE EXISTS (it does NOT replace probe_cv.py)
-------------------------------------------------------
probe_cv.py already fixed the first-order problem: it refits the
difference-of-means direction inside each fold and scores held-out data, so an
in-sample AUROC cannot masquerade as a representation. But it then does this:

    for L in range(n_layers):            # 19 layers
        v = cv_auroc(A[:, L], ...)       # keep the best
    nulls = [cv_auroc(A[:, bestL], ..., permute=True) for _ in range(perms)]

The reported AUROC is a MAXIMUM over 19 layers, while the null is computed at
the single already-selected layer. The null therefore describes "chance for one
fixed layer", not "chance for the best of 19". The comparison is between a
maximum and a mean, which biases every margin upward -- most dangerously for
the weak contrasts, where the whole question is whether a small margin is real.

This file computes the null the way the statistic was actually formed: for each
permutation, the labels are shuffled ONCE and the maximum CV AUROC over all
layers is taken. That null is the distribution of the same max statistic under
the null hypothesis, so the two are on the same footing. It reports

  auroc            best CV AUROC over layers            (as probe_cv.py)
  null_fixed       permutation mean at the chosen layer (as probe_cv.py, kept
                   so the two files can be reconciled line by line)
  null_selected    permutation mean of the MAX over layers  <-- the honest one
  null_selected_p95
  p_value          fraction of permutations whose max >= the observed value

A contrast is only called real if p_value is small against null_selected. Note
this correction can only ever move a result toward "not significant"; it cannot
manufacture one.

CHUNK MODE
----------
With --chunks, the stored action_chunk_pinned array (written by
probe_extract_pinned.py) is flattened per episode and pushed through the same
cross-validated pipeline. That reproduces behaviour_check.py's
"does the emitted chunk separate act from refuse" number, but with the
flow-matching noise pinned and with a selection-aware null.
"""
import argparse
import json
import pathlib

import numpy as np


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def cv_auroc(x, y, folds, rng, y_fixed=None):
    """Refit difference-of-means per fold; score held-out only.

    Identical arithmetic to probe_cv.cv_auroc. The one change is that label
    permutation is done by the CALLER (via y_fixed) rather than internally, so
    a single shuffled label vector can be reused across all layers -- which is
    what makes a max-over-layers null meaningful.
    """
    y = np.asarray(y if y_fixed is None else y_fixed)
    idx = rng.permutation(len(y))
    out = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if (y[tr].sum() < 2 or (~y[tr]).sum() < 2
                or y[te].sum() < 1 or (~y[te]).sum() < 1):
            continue
        d = x[tr][y[tr]].mean(0) - x[tr][~y[tr]].mean(0)
        nrm = np.linalg.norm(d)
        if nrm == 0:
            continue
        v = d / nrm
        s = x[te] @ v
        out.append(auroc(s[y[te]], s[~y[te]]))
    return float(np.nanmean(out)) if out else float("nan")


def evaluate(feats, y, folds, perms, seed=0):
    """feats: (n_views, n_samples, width). Returns the selection-aware summary.

    n_views is the number of layers for an activation stack, or 1 for a single
    feature matrix such as the emitted action chunk.
    """
    per_view = []
    for v in range(len(feats)):
        per_view.append(cv_auroc(feats[v], y, folds, np.random.default_rng(seed)))
    per_view = np.array(per_view, float)
    best = float(np.nanmax(per_view))
    best_v = int(np.nanargmax(per_view))

    # Null 1, as probe_cv.py computes it: permutations at the chosen view only.
    rng = np.random.default_rng(seed + 1)
    null_fixed = [cv_auroc(feats[best_v], y, folds, rng,
                           y_fixed=rng.permutation(y)) for _ in range(perms)]

    # Null 2, selection-aware: one shuffle, then the max over ALL views, which
    # is the same statistic as `best`.
    rng = np.random.default_rng(seed + 2)
    null_sel = []
    for _ in range(perms):
        yp = rng.permutation(y)
        vals = [cv_auroc(feats[v], y, folds, np.random.default_rng(seed),
                         y_fixed=yp) for v in range(len(feats))]
        null_sel.append(np.nanmax(vals))
    null_sel = np.array(null_sel, float)

    return {
        "auroc": best,
        "layer": best_v,
        "auroc_per_layer": per_view.tolist(),
        "null_fixed": float(np.nanmean(null_fixed)),
        "null_selected": float(np.nanmean(null_sel)),
        "null_selected_p95": float(np.nanpercentile(null_sel, 95)),
        "margin_vs_selected": best - float(np.nanmean(null_sel)),
        "p_value": float((null_sel >= best).sum() + 1) / (len(null_sel) + 1),
        "n_pos": int(np.sum(y)),
        "n_neg": int(np.sum(~np.asarray(y))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--chunks", action="store_true",
                    help="also score action_chunk_pinned if present")
    ap.add_argument("--out", default="probe_cv_strict.json")
    args = ap.parse_args()

    z = np.load(args.npz)
    payload = json.loads(pathlib.Path(args.meta).read_text())
    eps = payload["episodes"] if isinstance(payload, dict) else payload
    by = {int(e["episode_index"]): e for e in eps}
    rows = [by[int(i)] for i in z["episode_index"]]
    mode = np.array([r["mode"] for r in rows])
    band = np.array([r.get("band", "") for r in rows])
    rel = np.array([r.get("relation", "") for r in rows])
    act = mode == "act"

    contrasts = {
        "refusal      (refuse_hazard vs act)": (mode == "refuse_hazard", act),
        "ungrounded   (refuse_ungr.  vs act)": (mode == "refuse_ungrounded", act),
        "any_refusal  (both refusals vs act)": (mode != "act", act),
        "band  CONTROL(far vs near, acts)": (act & (band == "far"),
                                             act & (band == "near")),
        "relation CTRL(R1 vs R2, acts)": (act & (rel == "R1"),
                                          act & (rel == "R2")),
    }

    res = {}
    hdr = (f"{'contrast':38s} {'stack':7s} {'AUROC':>7s} {'nullFix':>8s} "
           f"{'nullSel':>8s} {'p95':>7s} {'p':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for name, (pos, neg) in contrasts.items():
        keep = pos | neg
        y = pos[keep]
        for stack in ("trunk", "expert"):
            key = f"{stack}_resid_mean"
            if key not in z:
                continue
            a = z[key][keep]                       # (n, NL+1, width)
            feats = np.transpose(a, (1, 0, 2))     # (NL+1, n, width)
            r = evaluate(feats, y, args.folds, args.perms)
            res[f"{name.strip()}|{stack}"] = r
            print(f"{name:38s} {stack:7s} {r['auroc']:7.3f} "
                  f"{r['null_fixed']:8.3f} {r['null_selected']:8.3f} "
                  f"{r['null_selected_p95']:7.3f} {r['p_value']:7.3f}",
                  flush=True)

        if args.chunks and "action_chunk_pinned" in z:
            c = z["action_chunk_pinned"][keep]
            feats = c.reshape(1, c.shape[0], -1)   # single view
            r = evaluate(feats, y, args.folds, args.perms)
            res[f"{name.strip()}|emitted_chunk"] = r
            print(f"{name:38s} {'CHUNK':7s} {r['auroc']:7.3f} "
                  f"{r['null_fixed']:8.3f} {r['null_selected']:8.3f} "
                  f"{r['null_selected_p95']:7.3f} {r['p_value']:7.3f}",
                  flush=True)

    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
