"""Preserve every 10k checkpoint of a FROZEN-TRUNK run without 95 GB of disk.

    # shrink one checkpoint to its trainable delta
    python checkpoint_slim.py slim --reference REF --checkpoint CKPT --out DIR

    # put it back together later
    python checkpoint_slim.py restore --reference REF --delta DIR --out CKPT

    # run alongside training: slim + prune as checkpoints appear
    python checkpoint_slim.py watch --reference REF --checkpoints DIR \
        --store DIR --keep_full 2 --until_step 100000

WHY THIS EXISTS
---------------
Each π0 checkpoint is 9.5 GB (8.3 GB of weights + 1.2 GB of optimizer state).
Saving one every 10 000 steps over a 100 000-step run is 95 GB. The Myriad
filesystem had 66 GB free at the time of writing — 863 GB of it belongs to a
different experiment that must not be touched — so a naive save_freq=10000
run would have died with a full disk somewhere around the seventh checkpoint,
taking the whole 25-hour job with it.

The saving grace is that this run trains with `train_expert_only=true`. The
PaliGemma trunk never receives a gradient, so it is bit-identical in every
checkpoint. Measured directly between the s1 050000 and 060000 checkpoints:

    173 of 778 tensors changed   0.64 GB
    605 of 778 identical         8.26 GB

So a checkpoint is fully described by a reference plus the 0.64 GB that moved.
Ten of those is 6.4 GB rather than 95 GB, and every 10k step remains
recoverable exactly — this is lossless, not a summary.

WHAT IS COMPARED. The delta is computed against a REFERENCE checkpoint by
actual tensor comparison, not by guessing which parameter names belong to the
expert. If the frozen trunk ever did move — a config slip, a different freeze
flag — those tensors would simply appear in the delta and the file would grow,
rather than being silently dropped. Size is the alarm.

WATCH MODE keeps the `--keep_full` most recent checkpoints intact, because
training resumes from `last` and deleting the newest would break the run. Only
older ones are reduced to a delta and removed. The `last` symlink is never
touched.
"""
import argparse
import json
import pathlib
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"


def _load(path):
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def slim(reference, checkpoint, out_dir):
    """Write only the tensors that differ from `reference`."""
    ref_w = pathlib.Path(reference) / WEIGHTS
    ckpt_w = pathlib.Path(checkpoint) / WEIGHTS
    if not ckpt_w.exists():
        raise SystemExit(f"no weights at {ckpt_w}")
    ref, cur = _load(ref_w), _load(ckpt_w)

    delta, kept, moved_bytes, same_bytes = {}, 0, 0, 0
    for k, v in cur.items():
        r = ref.get(k)
        nb = v.numel() * v.element_size()
        if r is None or r.shape != v.shape or not torch.equal(r, v):
            delta[k] = v
            moved_bytes += nb
        else:
            kept += 1
            same_bytes += nb

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(delta, str(out_dir / "expert_delta.safetensors"))

    # Everything except the weights is small and worth carrying verbatim, so a
    # restored checkpoint is loadable without hunting for its config.
    for name in ("config.json", "train_config.json", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        src = pathlib.Path(checkpoint) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    for extra in pathlib.Path(checkpoint).glob("*.safetensors"):
        if extra.name != WEIGHTS:              # normalizer/unnormalizer stats
            shutil.copy2(extra, out_dir / extra.name)

    manifest = {"reference": str(pathlib.Path(reference).resolve()),
                "checkpoint": str(pathlib.Path(checkpoint).resolve()),
                "changed_tensors": len(delta), "identical_tensors": kept,
                "changed_gb": round(moved_bytes / 1e9, 3),
                "identical_gb": round(same_bytes / 1e9, 3)}
    (out_dir / "delta_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  slim {pathlib.Path(checkpoint).name}: {len(delta)} changed "
          f"({moved_bytes/1e9:.2f} GB), {kept} identical "
          f"({same_bytes/1e9:.2f} GB) -> {out_dir}", flush=True)
    return manifest


def restore(reference, delta_dir, out_dir):
    """Rebuild a full checkpoint from reference + delta."""
    ref = _load(pathlib.Path(reference) / WEIGHTS)
    dl = _load(pathlib.Path(delta_dir) / "expert_delta.safetensors")
    ref.update(dl)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(ref, str(out_dir / WEIGHTS))
    for f in pathlib.Path(delta_dir).iterdir():
        if f.name not in ("expert_delta.safetensors", "delta_manifest.json"):
            shutil.copy2(f, out_dir / f.name)
    print(f"restored {len(ref)} tensors ({len(dl)} from delta) -> {out_dir}")


def watch(reference, ckpt_dir, store, keep_full, until_step, poll):
    """Slim and prune checkpoints as training produces them.

    The `keep_full` most recent are left untouched: training resumes from
    `last`, and removing the newest checkpoint would break the run.
    """
    ckpt_dir, store = pathlib.Path(ckpt_dir), pathlib.Path(store)
    store.mkdir(parents=True, exist_ok=True)
    done = set()
    print(f"[watch] {ckpt_dir} -> {store} (keep_full={keep_full}, "
          f"until step {until_step})", flush=True)
    while True:
        steps = sorted(int(p.name) for p in ckpt_dir.glob("[0-9]*")
                       if p.is_dir() and p.name.isdigit())
        # Never touch the newest keep_full, nor anything `last` points at.
        protect = set(steps[-keep_full:]) if steps else set()
        lastlink = ckpt_dir / "last"
        if lastlink.is_symlink():
            try:
                protect.add(int(pathlib.Path(lastlink).resolve().name))
            except ValueError:
                pass
        for s in steps:
            if s in protect or s in done:
                continue
            src = ckpt_dir / f"{s:06d}" / "pretrained_model"
            if not (src / WEIGHTS).exists():
                continue
            dest = store / f"{s:06d}"
            try:
                if not (dest / "expert_delta.safetensors").exists():
                    slim(reference, src, dest)
                # Only remove the full copy once the delta is on disk.
                if (dest / "expert_delta.safetensors").exists():
                    shutil.rmtree(ckpt_dir / f"{s:06d}")
                    print(f"  pruned full checkpoint {s:06d}", flush=True)
                done.add(s)
            except Exception as e:                     # never kill training
                print(f"  WARN slim/prune {s}: {type(e).__name__}: {e}",
                      flush=True)
        if steps and max(steps) >= until_step:
            print("[watch] final step reached; exiting", flush=True)
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("slim")
    s.add_argument("--reference", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)

    r = sub.add_parser("restore")
    r.add_argument("--reference", required=True)
    r.add_argument("--delta", required=True)
    r.add_argument("--out", required=True)

    w = sub.add_parser("watch")
    w.add_argument("--reference", required=True)
    w.add_argument("--checkpoints", required=True)
    w.add_argument("--store", required=True)
    w.add_argument("--keep_full", type=int, default=2)
    w.add_argument("--until_step", type=int, default=10 ** 9)
    w.add_argument("--poll", type=int, default=180)

    a = ap.parse_args()
    if a.cmd == "slim":
        slim(a.reference, a.checkpoint, a.out)
    elif a.cmd == "restore":
        restore(a.reference, a.delta, a.out)
    else:
        watch(a.reference, a.checkpoints, a.store, a.keep_full,
              a.until_step, a.poll)


if __name__ == "__main__":
    main()
