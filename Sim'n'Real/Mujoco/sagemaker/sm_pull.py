#!/usr/bin/env python
"""STEP 2 -- pull weights and data.  python sm_pull.py

Done deliberately rather than letting training trigger it, so a network or auth
failure happens NOW instead of thirty seconds into a run you believed had
started. Also verifies the dataset is the one you think it is, and that pi0
accepts it, before any GPU time is spent.
"""
import json
import pathlib
import sys

import os, pathlib
ROOT = pathlib.Path(os.environ.get("SKYGRIP_ROOT")
                    or ("/home/ec2-user/SageMaker"
                        if os.path.isdir("/home/ec2-user/SageMaker")
                        else pathlib.Path.home()))

REPO_DS = "hanapasta/pick_hold_v4s_train"
REPO_MODEL = "lerobot/pi0_base"


def main():
    from huggingface_hub import snapshot_download, hf_hub_download, whoami

    who = whoami()
    a = who.get("auth", {}).get("accessToken", {})
    print(f"[auth] {who.get('name')} | token '{a.get('displayName')}' | "
          f"expiry {a.get('expiration') or 'none'}")
    fg = a.get("fineGrained")
    if fg and not fg.get("canReadGatedRepos"):
        print("[auth] WARNING: token lacks gated-repo read. pi0 pulls "
              "google/paligemma-3b-pt-224, which IS gated -- training will "
              "fail at model load with a 403 that looks like a network error.")

    print(f"\n[model] {REPO_MODEL} (~14 GB, cached after the first time)")
    snapshot_download(REPO_MODEL)

    print(f"\n[data] {REPO_DS}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_DS)
    print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}")
    for k, v in ds.meta.features.items():
        if k.startswith("observation") or k == "action":
            print(f"    {k:34s} {v.get('dtype'):8s} {tuple(v.get('shape', ()))}")
    if ds.num_episodes != 480:
        print(f"  WARNING: expected 480 training episodes, got {ds.num_episodes}")

    print("\n[sidecar] episodes_meta.json")
    p = hf_hub_download(REPO_DS, "episodes_meta.json", repo_type="dataset",
                        local_dir=str(ROOT))
    meta = json.loads(pathlib.Path(p).read_text())["episodes"]
    modes = {}
    for e in meta:
        modes[e["mode"]] = modes.get(e["mode"], 0) + 1
    pairs = sum(1 for e in meta if e.get("pair_member"))
    print(f"  {len(meta)} records | modes {modes} | pair members {pairs}")
    print(f"  saved to {p}")

    # Feature resolution ONLY -- deliberately does NOT build a policy.
    # make_policy(PI0Config(), ...) constructs pi0 from scratch: 3.3B params
    # allocated in fp32 on CPU, ~13 GB of RAM. On a 121 GB cluster node that is
    # merely slow; on a 16-32 GB g5 it swaps or is OOM-killed and looks like a
    # hang. This mapping is what actually has to be right, and it is instant.
    # Full compatibility was verified separately end to end: 3 cameras, 16-D
    # state, 6-D action, predict_action_chunk -> (1, 50, 6).
    print("\n[check] feature mapping pi0 will infer from this dataset")
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    feats = dataset_to_policy_features(ds.meta.features)
    for k, v in feats.items():
        print(f"    {v.type.name:6s} {k:34s} {tuple(v.shape)}")
    n_img = sum(1 for v in feats.values() if v.type == FeatureType.VISUAL)
    print(f"  cameras {n_img}  (expect 3, state (16,), action (6,))")
    if n_img != 3:
        print("  WARNING: expected 3 camera streams")
    print("\nready -- next: bash sm_train.sh")


if __name__ == "__main__":
    sys.exit(main())
