"""Render N successful PICK episodes to a single review video.

Purpose is watching, not training: it uses a third-person camera that follows the
drone, so the whole task reads (approach, reach, grasp, carry, place) rather than the
onboard views. Failed attempts are skipped, so every episode in the video is a
success. Object colours/shapes and the approach style vary per episode.
"""
import argparse, io, contextlib
import numpy as np
import mujoco
import imageio.v2 as imageio
import collect_demos as C

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--out", type=str, default="../../Reports/pick_trials.mp4")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--reach_fraction", type=float, default=0.5)
# Camera. The default 1.5 m frames the whole task, which is what you want for
# reading the behaviour -- but it renders the gripper about 20 px across, so it is
# useless for checking the hardware. Drop to ~0.45 m and track the jaws instead of
# the body to inspect the grasp itself.
ap.add_argument("--distance", type=float, default=1.5,
                help="chase-camera distance, m (try 0.45 to inspect the gripper)")
ap.add_argument("--track", choices=["body", "jaws"], default="body",
                help="what the camera follows; 'jaws' keeps the grasp centred")
args = ap.parse_args()

C.EPISODE_MODE_P = [1.0, 0.0, 0.0]                 # picks only
C.REACH_FRACTION = args.reach_fraction             # half reach, half overhead
C.PHASE_FRAME_BUDGET["DESCEND"] = int(8.0 * C.FPS)

with contextlib.redirect_stdout(io.StringIO()):
    ctrl = C.SkyGripController(C.MODEL_PATH, C.MPPIParams(50, 16), flight="pd")
model, data = ctrl.model, ctrl.data
ik = C.ArmIK(model)
rend = mujoco.Renderer(model, height=480, width=640)
rng = np.random.default_rng(args.seed)

cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = args.distance, 105, -18

writer = imageio.get_writer(args.out, fps=C.FPS, codec="libx264",
                            quality=8, macro_block_size=1)
banked = attempts = 0
try:
    while banked < args.episodes and attempts < args.episodes * 4:
        attempts += 1
        frames = []
        def grab():
            # follow the drone so it never leaves frame as the table height varies,
            # or the jaws when the point is to watch the grasp
            cam.lookat[:] = (C.grasp_site_pos(model, data) if args.track == "jaws"
                             else data.qpos[0:3])
            rend.update_scene(data, camera=cam)
            frames.append(rend.render().copy())
        with contextlib.redirect_stdout(io.StringIO()):
            fr, info = C.run_episode(model, data, None, ctrl, ik, rng,
                                     "Pick up the {colour} {shape} and place it",
                                     on_frame=grab)
            ok, why = C.episode_succeeded(model, data, info)
        if not ok:
            print(f"  attempt {attempts}: discarded ({why})", flush=True)
            continue
        banked += 1
        style = "forward-reach" if info["reach_y"] is not None else "overhead"
        print(f"episode {banked}/{args.episodes}: {info['target_colour']} "
              f"{info['target_shape']:8s} | {style:13s} | {len(frames):4d} frames | {why}",
              flush=True)
        for f in frames:
            writer.append_data(f)
finally:
    writer.close()
print(f"\nwrote {args.out}  ({banked} episodes, {attempts} attempts)")
