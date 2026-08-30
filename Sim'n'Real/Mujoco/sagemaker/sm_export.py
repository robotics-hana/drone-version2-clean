#!/usr/bin/env python
"""STEP 6 -- get the finished checkpoint somewhere durable.

    python sm_export.py s1

The space's EBS is not a backup: if the space is deleted, so is the run. This
pushes the checkpoint to the Hub (PRIVATE) so the interpretability chain can
load it from anywhere -- including back on sparks.
"""
import pathlib
import sys

import os, pathlib
ROOT = pathlib.Path(os.environ.get("SKYGRIP_ROOT")
                    or ("/home/ec2-user/SageMaker"
                        if os.path.isdir("/home/ec2-user/SageMaker")
                        else pathlib.Path.home()))

TAG = sys.argv[1] if len(sys.argv) > 1 else "s1"
OUT = ROOT / f"pickhold_pi0_frozen_{TAG}"
REPO = f"hanapasta/pickhold-pi0-frozen-{TAG}"


def main():
    ck = OUT / "checkpoints" / "last" / "pretrained_model"
    if not ck.exists():
        cands = sorted((OUT / "checkpoints").glob("*/pretrained_model"))
        if not cands:
            raise SystemExit(f"no checkpoint under {OUT}")
        ck = cands[-1]
    print(f"[load] {ck}")

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(str(ck))

    n_all = sum(p.numel() for p in policy.parameters())
    n_tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[check] {n_all/1e9:.2f}B params, {n_tr/1e6:.0f}M trainable "
          f"({100*n_tr/n_all:.1f}%) -- expect ~10% for a frozen trunk")

    print(f"[push] {REPO} (private)")
    policy.push_to_hub(REPO, private=True)
    print("done\n")
    print("Next, the interpretability chain (see INTERPRETABILITY_README.md):")
    print(f"  python probe_extract_pi0.py --checkpoint {REPO} \\")
    print("      --dataset hanapasta/pick_hold_v4s_train \\")
    print("      --meta episodes_meta.json --out decision_activations.npz")
    print("  python direction_controls.py --npz decision_activations.npz \\")
    print("      --meta episodes_meta.json --out directions.npz")
    print("  python ablate_sweeps.py --sweep layer --direction refusal ...")


if __name__ == "__main__":
    main()
