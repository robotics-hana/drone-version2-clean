"""Closed-loop rollout of a trained pi0 policy in the v4style pick-and-hold
scene, with video, for all three episode modes.

    # 1. VALIDATE THE HARNESS FIRST (expert drives, policy never loaded)
    python rollout_v4style.py --driver expert --episodes 2 --out_dir vids/

    # 2. Only if the expert path scores successes, run the policy
    python rollout_v4style.py --driver policy --checkpoint CKPT --out_dir vids/

WHY THIS FILE EXISTS
--------------------
Every number in the interpretability analysis so far is OPEN-LOOP: a single
action chunk emitted at the decision frame and compared against the
demonstration. Nothing has ever executed this policy in the simulator, so the
question "does it actually pick anything up" has had no answer at all. The
three eval harnesses already in this directory (eval_pi0.py, eval_frozen_v1.py,
eval_frozen_v2.py) all target the separate AirVLA task -- they import
collect_airvla, fly to a wooden box through a gate, and emit a seven-channel
DELTA action -- so none of them can be pointed at this checkpoint. Hence a new
file. Nothing existing is modified: collect_demos.py is patched at runtime by
v4style.install(), exactly as the collector does it.

WHAT DRIVES THE PLANT
---------------------
The policy does not fly the drone directly. Actions in this dataset are
absolute SETPOINTS -- [drone_x, drone_y, drone_z, joint1, joint2, gripper] --
which the low-level controller tracks, and the training target was the
commanded setpoint rather than the realised thrust (see
collect_demos.SkyGripController.action). So a faithful rollout replaces only
the source of those setpoints:

    expert driver:  the phase planner sets them   (collect_demos.run_episode)
    policy driver:  policy.predict_action_chunk() sets them

and everything below that -- the smooth reference ramp, MPPI, the arm slew --
is identical in both. That is what makes the comparison meaningful.

THE EXPERT DRIVER IS NOT OPTIONAL
---------------------------------
A previous evaluation in this project returned 0/20 and the cause turned out
to be defects in the harness, not the policy. A rollout harness that has never
produced a success is indistinguishable from a policy that cannot succeed, so
--driver expert runs the identical scene, scoring and video path using
collect_demos.run_episode, whose behaviour is already validated by the 480
collected episodes. If the expert driver does not score successes here, the
harness is broken and any policy number it produces is meaningless. Run it
first, every time.

OBSERVATIONS
------------
The checkpoint's saved preprocessor contains a rename_observations_processor
mapping observation.images.camera1/2/3 to base_0_rgb / left_wrist_0_rgb /
right_wrist_0_rgb, so the RAW camera names are what must be fed here -- the
same keys the dataset carries, and the same ones probe_extract_pi0.py fed.
Images go in as float32 CHW in [0,1], matching what LeRobotDataset yields when
it decodes a video frame.

NOISE. pi0 is flow-matching and draws fresh unseeded noise per call, which was
the confound that invalidated an earlier ablation sweep. Here it is left
STOCHASTIC by default, because a rollout should show typical behaviour rather
than one lucky or unlucky draw; --noise_seed pins it when a reproducible
episode is wanted.
"""
import argparse
import json
import pathlib

import numpy as np
import mujoco

import collect_demos as C
import v4style as V

# Episode kinds accepted by v4style.draw_episode, keyed by the mode we want.
KIND = {
    "act": "act_tag",                  # act WITH a placard on a distractor:
                                       # the hardest act case, and the one that
                                       # makes "avoid placards" a losing rule
    "act_free": "act_free",            # act with no placard anywhere
    "refuse_hazard": "hazard",
    "refuse_ungrounded": "ungrounded",
}
VIEWS = ["camera3", "camera1", "camera2"]   # overview, wrist, scene
VIEW_TITLE = {"camera1": "wrist", "camera2": "scene", "camera3": "overview"}


def probe_state(model, data, action=None):
    """One row of the trajectory log.

    The question this exists to answer is "did the policy even TRY to reach the
    object", which the success flag cannot distinguish from "reached it and
    failed to hold it". The decisive quantity is the distance from the JAW
    MIDPOINT to the target object -- not the drone body, which can be directly
    above the object while the jaws are still 40 cm up. grasp_site_pos is the
    true grasp point (the body origin differs from it by 7.2 cm, wider than the
    object itself).
    """
    adr = model.jnt_qposadr[model.body_jntadr[model.body("target_object").id]]
    obj = np.array(data.qpos[adr:adr + 3])
    jaw = C.grasp_site_pos(model, data)
    row = {
        "drone": np.round(data.qpos[0:3], 4).tolist(),
        "joints": np.round(data.qpos[7:9], 4).tolist(),
        "jaw": np.round(jaw, 4).tolist(),
        "obj": np.round(obj, 4).tolist(),
        "jaw_obj_dist": round(float(np.linalg.norm(jaw - obj)), 4),
        "jaw_obj_xy": round(float(np.linalg.norm(jaw[0:2] - obj[0:2])), 4),
        "jaw_above_obj": round(float(jaw[2] - obj[2]), 4),
    }
    if action is not None:
        row["cmd"] = np.round(np.asarray(action, float), 4).tolist()
    return row


def build_obs(model, data, renderer, task_text, device):
    """One observation in exactly the form the checkpoint's preprocessor wants."""
    import torch
    obs = {}
    for key, cam in C.CAMERAS.items():
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()                      # HWC uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        obs[f"observation.images.{key}"] = t.unsqueeze(0).to(device)
    st = C.get_state(model, data)
    obs["observation.state"] = torch.from_numpy(st).float().unsqueeze(0).to(device)
    obs["task"] = [task_text]
    return obs


def tile(frames_by_view, banner_lines):
    """Compose the three camera views side by side with a text banner."""
    import cv2
    tiles = []
    for v in VIEWS:
        img = frames_by_view[v]
        small = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(small, VIEW_TITLE[v], (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(small)
    row = np.hstack(tiles)
    banner = np.zeros((26 * len(banner_lines) + 10, row.shape[1], 3), np.uint8)
    for i, line in enumerate(banner_lines):
        cv2.putText(banner, line, (8, 20 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([banner, row])


def draw(rng, mode):
    """Choose the episode content and publish it to v4style's state, which is
    what the patched choose_episode/randomise_episode read. Must happen before
    the scene is randomised, by either driver."""
    ep, tags, meta = V.draw_episode(rng, KIND[mode])
    V._STATE.update({"ep": ep, "tags": tags, "spawn_mode": "train",
                     "tag_variant": None, "band": meta["band"]})
    return ep, meta


def setup_episode(model, data, ctrl, rng, mode):
    """Replicates the setup block of collect_demos.run_episode, so the scene the
    policy sees is byte-for-byte the scene the collector would have built.

    Used ONLY by the policy driver. The expert driver must not call this,
    because collect_demos.run_episode performs this same setup internally and
    would randomise the scene a second time -- leaving the info dict used for
    scoring describing a scene that no longer exists.
    """
    # These are the patched versions installed by V.install().
    ep_c = C.choose_episode(rng, mode=None)
    colour_t, shape_t = ep_c["target"]
    colour_d, shape_d = ep_c["distractor"]
    colour_d2, shape_d2 = ep_c["distractor2"]
    scene = C.randomise_episode(model, data, rng, colour_t, shape_t,
                                colour_d, shape_d, colour_d2, shape_d2)
    ctrl.reset_after_randomisation()
    task_text = V.TASK.format(colour=ep_c["named_colour"],
                              shape=ep_c["named_shape"])
    info = {"place": scene["place"].copy(), "obj_start": scene["target"].copy(),
            "obj_half_height": scene["target_half_h"], "reach_y": None,
            "task": task_text, "mode": ep_c["mode"],
            "named_colour": ep_c["named_colour"],
            "named_shape": ep_c["named_shape"], "is_pick": ep_c["is_pick"],
            "target_colour": scene["target_colour"],
            "target_shape": scene["target_shape"],
            "distractor_colour": scene["distractor_colour"],
            "distractor_start": scene["distractor"].copy(),
            "distractor2_colour": scene["distractor2_colour"],
            "distractor2_start": scene["distractor2"].copy(),
            "surface_z": scene["surface_z"],
            "safe_z": scene["surface_z"] + C.SAFE_HEIGHT}
    return ep_c, scene, task_text, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=["expert", "policy"], required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="required for --driver policy")
    ap.add_argument("--modes", default="act,refuse_hazard,refuse_ungrounded")
    ap.add_argument("--episodes", type=int, default=1, help="per mode")
    ap.add_argument("--exec_horizon", type=int, default=25,
                    help="frames executed from each predicted chunk before "
                         "replanning; the chunk is 50 long")
    ap.add_argument("--max_frames", type=int, default=450)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="pin the flow-matching noise for reproducibility")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdframes", type=int, default=5)
    ap.add_argument("--graspcap", type=float, default=2.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    V.tune(hold_frames=args.holdframes, grasp_cap_s=args.graspcap)
    V.install()
    ctrl = V.build_controller()
    model, data = ctrl.model, ctrl.data
    ik = C.ArmIK(model)
    renderer = mujoco.Renderer(model, height=C.IMG_H, width=C.IMG_W)
    substeps = max(1, round((1.0 / C.FPS) / model.opt.timestep))
    print(f"[setup] {substeps} physics substeps per frame, "
          f"driver={args.driver}", flush=True)

    policy = pre = post = None
    device = "cpu"
    if args.driver == "policy":
        if not args.checkpoint:
            raise SystemExit("--driver policy requires --checkpoint")
        import torch
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.factory import make_pre_post_processors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = PI0Policy.from_pretrained(args.checkpoint).to(device).eval()
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}})
        print(f"[setup] policy on {device}", flush=True)

    results = []
    modes = args.modes.split(",")
    for mi, mode in enumerate(modes):
        for k in range(args.episodes):
            # Deterministic per (mode, episode). Python's hash() of a str is
            # salted per process, so using it here would make --seed useless
            # for reproducing a specific episode.
            seed = args.seed * 1000 + mi * 101 + k
            rng = np.random.default_rng(seed)
            ep, meta = draw(rng, mode)
            task_text = V.TASK.format(colour=ep["named_colour"],
                                      shape=ep["named_shape"])
            info = None
            if args.driver == "policy":
                ep, scene, task_text, info = setup_episode(model, data, ctrl,
                                                           rng, mode)
            vid = []

            def snap(extra_lines):
                by_view = {}
                for key, cam in C.CAMERAS.items():
                    renderer.update_scene(data, camera=cam)
                    by_view[key] = renderer.render().copy()
                vid.append(tile(by_view, extra_lines))

            traj = []
            if args.driver == "expert":
                # The proven path: collect_demos drives, we only record video.
                def on_frame():
                    traj.append(probe_state(model, data))
                    snap([f"EXPERT  |  {mode}",
                          f'"{task_text}"'])
                # run_episode does its own choose/randomise, so its returned
                # info is the authoritative description of the scene that was
                # actually flown. Score against that, not against a copy.
                fr, info = C.run_episode(model, data, None, ctrl, ik, rng,
                                         V.TASK, mode=ep["mode"],
                                         on_frame=on_frame)
                task_text = info["task"]
                n_frames = len(fr)
            else:
                import torch
                chunk = None
                noise = None
                if args.noise_seed is not None:
                    g = torch.Generator(device="cpu").manual_seed(args.noise_seed)
                    noise = torch.normal(
                        0.0, 1.0,
                        size=(1, policy.config.chunk_size,
                              policy.config.max_action_dim),
                        generator=g).to(device)
                n_frames = 0
                for f in range(args.max_frames):
                    if f % args.exec_horizon == 0:
                        obs = build_obs(model, data, renderer, task_text, device)
                        with torch.no_grad():
                            kw = {"noise": noise} if noise is not None else {}
                            ch = policy.predict_action_chunk(pre(obs), **kw)
                            ch = post(ch)
                        chunk = ch[0].float().cpu().numpy()
                    a = chunk[f % args.exec_horizon]
                    ctrl.set_targets(a[0:3], a[3:5], float(a[5]))
                    for _ in range(substeps):
                        ctrl.step()
                        mujoco.mj_step(model, data)
                    n_frames += 1
                    traj.append(probe_state(model, data, a))
                    snap([f"POLICY  |  {mode}",
                          f'"{task_text}"',
                          f"cmd xyz {a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}  "
                          f"j1 {a[3]:+.2f}  j2 {a[4]:+.2f}  grip {a[5]:.3f}",
                          f"jaw-obj {traj[-1]['jaw_obj_dist']*1000:.0f} mm "
                          f"(xy {traj[-1]['jaw_obj_xy']*1000:.0f}, "
                          f"dz {traj[-1]['jaw_above_obj']*1000:+.0f})"])
                    if data.qpos[2] < info["surface_z"] + 0.05:
                        print("   crashed, stopping episode early", flush=True)
                        break

            ok, why = V.hold_succeeded(model, data, info)
            name = f"{args.driver}_{mode}_{k}"
            path = out / f"{name}.mp4"
            imageio.mimsave(path, vid, fps=args.fps, quality=7)
            (out / f"{name}_traj.json").write_text(json.dumps(traj))

            # The closest the jaws ever came to the object, which separates
            # "tried and missed" from "never approached".
            closest = min((r["jaw_obj_dist"] for r in traj), default=float("nan"))
            closest_xy = min((r["jaw_obj_xy"] for r in traj), default=float("nan"))
            grips = [r["cmd"][5] for r in traj if "cmd" in r]
            print(f"[{name}] {'SUCCESS' if ok else 'FAIL'} :: {why} "
                  f"({n_frames} frames) | closest jaw-obj {closest*1000:.0f} mm "
                  f"(xy {closest_xy*1000:.0f} mm)"
                  + (f" | grip cmd {min(grips):.4f}..{max(grips):.4f}"
                     if grips else "")
                  + f" -> {path}", flush=True)
            results.append({"driver": args.driver, "mode": mode, "episode": k,
                            "seed": seed, "task": task_text,
                            "success": bool(ok), "why": why,
                            "n_frames": int(n_frames), "video": str(path),
                            "closest_jaw_obj_m": closest,
                            "closest_jaw_obj_xy_m": closest_xy,
                            "grip_cmd_min": (min(grips) if grips else None),
                            "grip_cmd_max": (max(grips) if grips else None)})

    summ = out / f"rollout_{args.driver}.json"
    summ.write_text(json.dumps(results, indent=1))
    n_ok = sum(r["success"] for r in results)
    print(f"\n[{args.driver}] {n_ok}/{len(results)} succeeded -> {summ}")


if __name__ == "__main__":
    main()
