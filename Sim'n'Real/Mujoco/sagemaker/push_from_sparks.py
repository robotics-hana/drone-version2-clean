#!/usr/bin/env python
"""STEP 0 -- run this ON SPARKS, once. Publishes the datasets to the Hub.

    ~/skyenv/bin/python push_from_sparks.py

SageMaker cannot reach sparks (it sits behind a cloudflared tunnel), so the Hub
is the transfer path. ~2.9 GB total.

TWO THINGS THIS GETS RIGHT THAT A PLAIN push_to_hub DOES NOT:
  * private=True. LeRobotDataset.push_to_hub defaults to private=None, which
    creates PUBLIC repos. Research data should not become public by omission.
  * the sidecar. push_to_hub carries the dataset but NOT episodes_meta.json,
    and every interpretability script depends on it (decision_frame, relation,
    band, pair fingerprints). It is uploaded separately here.
"""
import sys
from pathlib import Path

from huggingface_hub import upload_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SPLITS = ("train", "eval", "recover")
OWNER = "hanapasta"


def main():
    only = sys.argv[1:] or SPLITS
    for split in only:
        repo = f"{OWNER}/pick_hold_v4s_{split}"
        print(f"\n=== {repo} ===", flush=True)
        ds = LeRobotDataset(repo)
        print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames", flush=True)

        ds.push_to_hub(private=True, card_kwargs={})
        print("  dataset pushed (private)", flush=True)

        sidecar = Path(ds.root) / "episodes_meta.json"
        if sidecar.exists():
            upload_file(path_or_fileobj=str(sidecar),
                        path_in_repo="episodes_meta.json",
                        repo_id=repo, repo_type="dataset")
            print("  episodes_meta.json pushed", flush=True)
        else:
            print("  WARNING: no episodes_meta.json -- run install_sidecar.py "
                  "first, or the interpretability scripts will have no "
                  "decision frames to work from", flush=True)
    print("\ndone")


if __name__ == "__main__":
    main()
