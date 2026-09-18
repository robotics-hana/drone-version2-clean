"""Per-channel act-vs-refuse profile, split by REFUSAL TYPE.

    python channel_dissociation.py --root DROOT --meta episodes_meta.json \
        [--npz decision_activations_pinned.npz] --out channel_dissociation.json

WHY THIS FILE EXISTS (it does NOT replace behaviour_check.py)
--------------------------------------------------------------
behaviour_check.py compares the policy's emitted chunk against the expert
demonstration for ONE contrast: act vs refuse_hazard. Three separate results
now point at that being the wrong granularity:

  * probe_cv.py: on the expert stack, refuse_hazard vs act sits at chance
    (0.551, null 0.496) while refuse_ungrounded vs act reaches 0.684
    (null 0.474).
  * direction_controls.py: on the expert stack those two refusal directions
    are nearly PARALLEL -- mean cosine +0.934, max 0.947. So there is one
    refusal axis, not two, and the hazard population simply barely moves
    along it.
  * behaviour_check.py: on the hazard contrast, joint2 separates by -0.122 in
    the policy against -0.836 in the demonstrations.

Together those say the interesting quantity is not "did it learn to refuse"
but "which refusal did it learn". The two refusal types differ in what has to
be computed to trigger them:

  refuse_ungrounded  the named colour is absent from the scene entirely, so a
                     language/vision mismatch is sufficient.
  refuse_hazard      the named object is present but placarded, so the placard
                     has to be BOUND to the named object -- a distractor may
                     carry a placard without licensing a refusal.

This file measures both against the same expert demonstrations, per channel,
so the dissociation can be quantified rather than inferred.

WHAT IT REPORTS, per channel and per refusal mode

  demo_delta    mean(refuse) - mean(act) in the DEMONSTRATIONS. This is the
                training target: how far apart the expert put the two.
  policy_delta  the same difference in the policy's emitted chunks.
  expression    policy_delta / demo_delta. 1.0 = the policy reproduces the
                demonstrated separation; 0.0 = it ignores it. Only meaningful
                where |demo_delta| is large enough to divide by, so channels
                below --min_delta are reported with expression = null rather
                than a ratio blown up by a near-zero denominator.

Demonstrations are read from the parquet rather than by indexing the dataset,
because indexing decodes three video streams per frame (the mistake that made
behaviour_check.py 50x slower than the forward pass it was timing).

The policy half is optional: pass --npz pointing at the output of
probe_extract_pinned.py, which stores action_chunk_pinned for every episode.
Those chunks were produced with the flow-matching noise PINNED, so they do not
carry the sampling variance that unpinned chunks do. Without --npz the script
reports the demonstration profile alone, which needs no GPU and no policy.
"""
import argparse
import glob
import json
import pathlib

import numpy as np

CHANNELS = ["x", "y", "z", "joint1", "joint2", "gripper"]
MODES = ["act", "refuse_hazard", "refuse_ungrounded"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--npz", default=None,
                    help="npz with action_chunk_pinned (optional)")
    ap.add_argument("--horizon", type=int, default=50,
                    help="action-chunk horizon to average over")
    ap.add_argument("--min_delta", type=float, default=0.05,
                    help="|demo_delta| below this reports expression as null")
    ap.add_argument("--out", default="channel_dissociation.json")
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.meta).read_text())
    eps = payload["episodes"] if isinstance(payload, dict) else payload
    eps = [e for e in eps if e.get("decision_frame", -1) >= 0]

    # --- demonstrations, straight from the parquet -------------------------
    import pandas as pd
    files = sorted(glob.glob(str(pathlib.Path(args.root) / "data" / "**" / "*.parquet"),
                             recursive=True))
    tab = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index",
                                                 "action"])
                     for f in files], ignore_index=True)
    by_ep = {}
    for ep, grp in tab.sort_values("frame_index").groupby("episode_index"):
        by_ep[int(ep)] = np.stack(grp["action"].to_numpy())
    print(f"demonstrations loaded for {len(by_ep)} episodes", flush=True)

    def demo_window(e):
        """Expert actions over the chunk horizon starting at the decision
        frame, padded by repeating the last action if the episode ends."""
        a = by_ep.get(int(e["episode_index"]))
        if a is None:
            return None
        d0 = int(e["decision_frame"])
        w = a[d0:d0 + args.horizon]
        if len(w) == 0:
            return None
        if len(w) < args.horizon:
            w = np.concatenate([w, np.repeat(w[-1:], args.horizon - len(w), 0)])
        return w

    demo_mean = {}
    counts = {}
    for m in MODES:
        rows = [e for e in eps if e["mode"] == m]
        w = [demo_window(e) for e in rows]
        w = [x for x in w if x is not None]
        counts[m] = len(w)
        demo_mean[m] = np.stack(w).mean(axis=(0, 1)) if w else None
        print(f"  demo {m:18s} n={len(w)}", flush=True)

    # --- policy chunks, if a pinned npz was supplied -----------------------
    pol_mean = None
    if args.npz and pathlib.Path(args.npz).exists():
        z = np.load(args.npz)
        if "action_chunk_pinned" in z:
            idx = {int(i): r for r, i in enumerate(z["episode_index"])}
            chunks = z["action_chunk_pinned"]           # (n, H, adim)
            pol_mean, pol_counts = {}, {}
            for m in MODES:
                rows = [idx[int(e["episode_index"])] for e in eps
                        if e["mode"] == m and int(e["episode_index"]) in idx]
                pol_counts[m] = len(rows)
                pol_mean[m] = (chunks[rows].mean(axis=(0, 1)) if rows else None)
                print(f"  policy {m:18s} n={len(rows)}", flush=True)
            counts = {"demo": counts, "policy": pol_counts}
        else:
            print("  (npz has no action_chunk_pinned; demonstrations only)")

    # --- assemble ----------------------------------------------------------
    out = {"horizon": args.horizon, "counts": counts, "channels": {}}
    hdr = f"{'channel':9s} {'mode':18s} {'demo Δ':>9s} {'policy Δ':>10s} {'expr':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for ci, name in enumerate(CHANNELS):
        entry = {"demo_act": float(demo_mean["act"][ci])}
        if pol_mean and pol_mean["act"] is not None:
            entry["policy_act"] = float(pol_mean["act"][ci])
        for m in ("refuse_hazard", "refuse_ungrounded"):
            if demo_mean[m] is None:
                continue
            dd = float(demo_mean[m][ci] - demo_mean["act"][ci])
            rec = {"demo_refuse": float(demo_mean[m][ci]), "demo_delta": dd}
            pd_ = None
            if pol_mean and pol_mean[m] is not None:
                pd_ = float(pol_mean[m][ci] - pol_mean["act"][ci])
                rec["policy_refuse"] = float(pol_mean[m][ci])
                rec["policy_delta"] = pd_
                rec["expression"] = (pd_ / dd if abs(dd) >= args.min_delta
                                     else None)
            entry[m] = rec
            ex = rec.get("expression")
            print(f"{name:9s} {m:18s} {dd:+9.4f} "
                  f"{(f'{pd_:+10.4f}' if pd_ is not None else '         -')} "
                  f"{(f'{ex:7.3f}' if ex is not None else '      -')}")
        out["channels"][name] = entry

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
