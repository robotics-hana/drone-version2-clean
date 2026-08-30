"""Collect the v4-style dataset to LeRobotDataset, per the V4_SPEC allocation.

    python collect_v4style.py --split train   --out_repo_id hanapasta/pick_hold_v4s_train
    python collect_v4style.py --split eval    --out_repo_id hanapasta/pick_hold_v4s_eval
    python collect_v4style.py --split recover --out_repo_id hanapasta/pick_hold_v4s_recover

Episode CONTENT and the flight itself come from v4style.py, which is shared
with render_v4style_demo.py so the data and the review video cannot diverge.
This file adds only the schedule, the LeRobot writing, and the metadata
sidecar.

ALLOCATION (730 episodes total)
  train   480 = 120 matched pairs (240 eps) + 24 act w/ placard on distractor
                + 96 placard-free act + 120 refuse-ungrounded
                -> 240 act / 120 refuse_hazard / 120 refuse_ungrounded (50/25/25)
                -> P(placard visible | act) = (120+24)/240 = 0.60, so
                   "approach the unplacarded object" solves only 60% of acts
                   and is not a viable policy
  eval    160 = 40 novel colour (orange/purple, never trained)
                + 40 novel spawn (start outside the object row)
                + 30 held-out matched pairs (60 eps) + 20 varied placard look
  recover  90 = 45 act / 22 refuse_hazard / 23 refuse_ungrounded, spawned at
                the policy's own stuck states. Mode-balanced EXACTLY like the
                base mix so recovery context carries zero information about the
                decision.

THE SIDECAR (<repo>_meta.json) is not optional bookkeeping -- the
interpretability work depends on it. LeRobot frames carry only the task
string, but direction extraction has to select populations by mode, relation
and band, and has to know each episode's DECISION FRAME (the last frame of the
shared dwell, where both populations have flown identical trajectories and
differ only in what they are about to do). Both are recorded here per episode.

MATCHED PAIRS are banked ATOMICALLY: both members succeed and their scene
fingerprints match, or neither is written. A half-pair in the data would
silently break every paired analysis downstream.
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
import mujoco

import collect_demos as C
import v4style as V

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=("train", "eval", "recover"), required=True)
ap.add_argument("--out_repo_id", type=str, default=None)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--limit", type=int, default=0,
                help="collect only the first N slots (smoke runs)")
ap.add_argument("--max_attempts", type=int, default=6,
                help="retries per slot before it is abandoned")
ap.add_argument("--no_save", action="store_true",
                help="fly everything, write nothing (setup check)")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--holdframes", type=int, default=5)
ap.add_argument("--graspcap", type=float, default=2.0)
args = ap.parse_args()

if not args.no_save and not args.out_repo_id:
    ap.error("--out_repo_id is required unless --no_save is given")


def build_schedule(split, rng):
    """The V4_SPEC allocation. A 'pair' slot carries BOTH members."""
    slots = []
    if split == "train":
        for k in range(120):
            slots.append({"kind": "pair", "pair_id": k, "spawn": "train"})
        for _ in range(24):
            slots.append({"kind": "act_tag", "spawn": "train"})
        for _ in range(96):
            slots.append({"kind": "act_free", "spawn": "train"})
        for _ in range(120):
            slots.append({"kind": "ungrounded", "spawn": "train"})
    elif split == "eval":
        for _ in range(40):
            slots.append({"kind": "novel_colour", "spawn": "train"})
        for _ in range(40):
            slots.append({"kind": "novel_spawn", "spawn": "novel"})
        for k in range(30):
            slots.append({"kind": "eval_pair", "pair_id": 500 + k,
                          "spawn": "train"})
        for k in range(20):
            slots.append({"kind": "varied_tag", "spawn": "train",
                          "variant": k % len(V.TAG_VARIANTS)})
    else:  # recover -- mode-balanced like the base mix, at stuck-state spawns
        for k in range(45):                       # act, 60% placarded
            slots.append({"kind": "act_tag" if k < 27 else "act_free",
                          "spawn": "recover"})
        for _ in range(22):
            slots.append({"kind": "hazard", "spawn": "recover"})
        for _ in range(23):
            slots.append({"kind": "ungrounded", "spawn": "recover"})
    rng.shuffle(slots)
    return slots


V.tune(hold_frames=args.holdframes, grasp_cap_s=args.graspcap)
V.install()
ctrl = V.build_controller()
model, data = ctrl.model, ctrl.data
ik = C.ArmIK(model)
renderer = None if args.no_save else mujoco.Renderer(model, height=C.IMG_H,
                                                     width=C.IMG_W)
seg = mujoco.Renderer(model, height=240, width=320)
seg.enable_segmentation_rendering()
G = int(mujoco.mjtObj.mjOBJ_GEOM)


def placard_px():
    """Placard area in the wrist camera right now -- the evidence check. A
    refuse_hazard episode whose placard is 0 px at the decision frame is not a
    demonstration of refusing, it is a demonstration of guessing."""
    seg.update_scene(data, camera="wrist_cam")
    s = seg.render()
    ids = [model.geom(g).id for g in V.HAZARD_GEOMS.values()
           if model.geom_rgba[model.geom(g).id, 3] > 0.5]
    return int(sum(((s[:, :, 1] == G) & (s[:, :, 0] == g)).sum() for g in ids))


def fly(ep, tags, meta, seed, spawn_mode, variant=None):
    """Run one episode. Returns (frames, ok, why, record)."""
    V._STATE.update({"ep": ep, "tags": tags, "spawn_mode": spawn_mode,
                     "tag_variant": variant, "band": meta["band"]})
    rng = np.random.default_rng(seed)
    dwell = int(rng.integers(V.DWELL_RANGE[0], V.DWELL_RANGE[1] + 1))
    C.PHASE_FRAME_BUDGET["HOLD"] = dwell
    frames, dwell_idx, dwell_px = [], [], []

    def on_frame():
        i = len(frames)
        frames.append(i)
        wp = V._STATE.get("inspect_wp")
        if wp is not None and ctrl.phase_goal is not None \
                and np.allclose(ctrl.phase_goal, wp):
            dwell_idx.append(i)
            dwell_px.append(placard_px())

    fr, info = C.run_episode(model, data, renderer, ctrl, ik, rng, V.TASK,
                             mode=ep["mode"], on_frame=on_frame)
    ok, why = V.hold_succeeded(model, data, info)
    rec = dict(meta)
    rec.update({
        "mode": ep["mode"], "is_pick": bool(ep["is_pick"]),
        "task": info["task"], "spawn_mode": spawn_mode,
        "named_colour": ep["named_colour"], "named_shape": ep["named_shape"],
        "target": list(ep["target"]), "distractor": list(ep["distractor"]),
        "distractor2": list(ep["distractor2"]),
        "tags": {k: bool(v) for k, v in tags.items()},
        "tag_variant": variant,
        "n_frames": len(fr),
        # DECISION FRAME: last frame of the shared dwell. Everything before it
        # is identical in distribution between act and refuse.
        "decision_frame": (dwell_idx[-1] if dwell_idx else -1),
        "dwell_frames": dwell,
        "placard_px_at_decision": (dwell_px[-1] if dwell_px else 0),
        "fingerprint": V._STATE["fingerprint"],
        "seed": seed, "success": bool(ok), "why": why,
    })
    return fr, ok, why, rec


# --- dataset ---------------------------------------------------------------
dataset = None
if not args.no_save:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if args.overwrite:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root = pathlib.Path(HF_LEROBOT_HOME) / args.out_repo_id
        if root.exists():
            shutil.rmtree(root)
            print(f"removed existing dataset at {root}", flush=True)
    features = {
        f"observation.images.{k}": {"dtype": "video",
                                    "shape": (C.IMG_H, C.IMG_W, 3),
                                    "names": ["height", "width", "channels"]}
        for k in C.CAMERAS
    }
    features["observation.state"] = {"dtype": "float32",
                                     "shape": (len(C.STATE_NAMES),),
                                     "names": C.STATE_NAMES}
    features["action"] = {"dtype": "float32",
                          "shape": (len(C.ACTION_NAMES),),
                          "names": C.ACTION_NAMES}
    dataset = LeRobotDataset.create(repo_id=args.out_repo_id, fps=C.FPS,
                                    features=features, robot_type="skygrip",
                                    rgb_encoder=C.pick_rgb_encoder())

rng = np.random.default_rng(args.seed)
schedule = build_schedule(args.split, rng)
if args.limit:
    schedule = schedule[:args.limit]

records, banked, attempts_total, abandoned = [], 0, 0, 0
seed_ctr = args.seed * 1000


def bank(frames, rec):
    global banked
    if dataset is not None:
        for f in frames:
            dataset.add_frame(f)
        dataset.save_episode(parallel_encoding=False)
    rec["episode_index"] = banked
    records.append(rec)
    banked += 1


def collect():
  global seed_ctr, attempts_total, abandoned
  for si, slot in enumerate(schedule):
      kind, spawn_mode = slot["kind"], slot["spawn"]
      variant = slot.get("variant")
      done = False
      for attempt in range(args.max_attempts):
          seed_ctr += 1
          attempts_total += 1
          if kind in ("pair", "eval_pair"):
              # Atomic: draw both members from the SAME seed, require both to
              # succeed AND their fingerprints to match, then bank together.
              out = {}
              for m in ("A", "B"):
                  r = np.random.default_rng(seed_ctr)
                  ep, tags, meta = V.draw_episode(r, kind, member=m)
                  meta["pair_id"] = slot["pair_id"]
                  meta["pair_member"] = m
                  out[m] = fly(ep, tags, meta, seed_ctr, spawn_mode, variant)
              okA, okB = out["A"][1], out["B"][1]
              fpA, fpB = out["A"][3]["fingerprint"], out["B"][3]["fingerprint"]
              if okA and okB and fpA == fpB:
                  for m in ("A", "B"):
                      out[m][3]["pair_fingerprint_match"] = True
                      bank(out[m][0], out[m][3])
                  done = True
          else:
              r = np.random.default_rng(seed_ctr)
              ep, tags, meta = V.draw_episode(r, kind)
              frames, ok, why, rec = fly(ep, tags, meta, seed_ctr, spawn_mode,
                                         variant)
              if ok:
                  bank(frames, rec)
                  done = True
          if done:
              break
      if not done:
          abandoned += 1
          print(f"  slot {si} ({kind}): abandoned after {args.max_attempts} "
                f"attempts", flush=True)
      if (si + 1) % 10 == 0 or si == len(schedule) - 1:
          print(f"slot {si+1}/{len(schedule)} | banked {banked} eps | "
                f"{attempts_total} attempts | {abandoned} abandoned", flush=True)
          if args.out_repo_id:
              meta_path = pathlib.Path(f"{args.out_repo_id.split('/')[-1]}"
                                       f"_meta.json")
              meta_path.write_text(json.dumps(
                  {"split": args.split, "episodes": records}, indent=1))

try:
    collect()
finally:
    if dataset is not None:
        # ALWAYS finalise: a run killed mid-way otherwise leaves a
        # dataset with no parquet footer, i.e. unreadable.
        dataset.finalize()
        print(f"finalised {args.out_repo_id} ({banked} episodes)")

# --- summary ---------------------------------------------------------------
by_mode, by_rel, by_band, by_kind = {}, {}, {}, {}
for r in records:
    by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
    by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    by_rel[r["relation"]] = by_rel.get(r["relation"], 0) + 1
    by_band[r["band"]] = by_band.get(r["band"], 0) + 1
acts = [r for r in records if r["is_pick"]]
tagged_acts = [r for r in acts if any(r["tags"].values())]
haz = [r for r in records if r["mode"] == "refuse_hazard"]
print(f"\n=== {args.split}: banked {banked} episodes in {attempts_total} "
      f"attempts ({abandoned} slots abandoned) ===")
print(f"by mode  : {by_mode}")
print(f"by kind  : {by_kind}")
print(f"relation : {by_rel}")
print(f"band     : {by_band}")
if acts:
    print(f"P(placard | act) = {len(tagged_acts)}/{len(acts)} = "
          f"{len(tagged_acts)/len(acts):.2f}  (spec 0.60)")
if haz:
    px = [r["placard_px_at_decision"] for r in haz]
    print(f"refuse_hazard placard px at decision: min {min(px)} "
          f"median {int(np.median(px))} max {max(px)}")
if args.out_repo_id:
    meta_path = pathlib.Path(f"{args.out_repo_id.split('/')[-1]}_meta.json")
    meta_path.write_text(json.dumps({"split": args.split,
                                     "episodes": records}, indent=1))
    print(f"wrote {meta_path}")

