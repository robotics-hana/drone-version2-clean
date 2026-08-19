"""Render pick-and-place episodes flown by the MPPI flight controller.

Same task FSM and scene randomisation as render_trials.py, but the body is
flown by PureMPPIController (create_force_general.py) instead of the PD
controller. Unlike render_trials.py, EVERY attempt is written to the video,
successful or not -- the point is to demonstrate what MPPI does, and the
repo's own measurements (106 mm RMS station keeping vs the ~10 mm the grasp
needs) predict most attempts will miss, so a success-only filter could
discard the entire run.

Run from this directory:
    python mppi_demo.py --episodes 2 --out ../../Reports/mppi_pick_trial.mp4
"""
import argparse, io, contextlib, time
import numpy as np
import mujoco
import imageio.v2 as imageio
import collect_demos as C

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--out", type=str, default="../../Reports/mppi_pick_trial.mp4")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--samples", type=int, default=50)
ap.add_argument("--horizon", type=int, default=16)
ap.add_argument("--distance", type=float, default=1.5)
args = ap.parse_args()

C.EPISODE_MODE_P = [1.0, 0.0, 0.0]        # picks only
# Overhead approach only: the forward-reach gesture was tuned on the PD
# controller and slings the arm CoM out on a lever, which is exactly what the
# repo notes crashed MPPI. Give MPPI its best case.
C.REACH_FRACTION = 0.0
C.PHASE_FRAME_BUDGET["DESCEND"] = int(8.0 * C.FPS)

ctrl = C.SkyGripController(
    C.MODEL_PATH,
    C.MPPIParams(num_samples=args.samples, horizon=args.horizon),
    flight="mppi")
model, data = ctrl.model, ctrl.data
ik = C.ArmIK(model)
rend = mujoco.Renderer(model, height=480, width=640)
rng = np.random.default_rng(args.seed)

cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = args.distance, 105, -18

writer = imageio.get_writer(args.out, fps=C.FPS, codec="libx264",
                            quality=8, macro_block_size=1)
results = []
t0 = time.time()
try:
    for ep in range(1, args.episodes + 1):
        frames = []
        def grab():
            cam.lookat[:] = data.qpos[0:3]      # follow the drone
            rend.update_scene(data, camera=cam)
            frames.append(rend.render().copy())
        print(f"--- episode {ep}/{args.episodes} (MPPI, {args.samples} samples, "
              f"horizon {args.horizon}) ---", flush=True)
        fr, info = C.run_episode(model, data, None, ctrl, ik, rng,
                                 "Pick up the {colour} {shape} and place it",
                                 verbose=True, on_frame=grab)
        ok, why = C.episode_succeeded(model, data, info)
        results.append((ok, why, info))
        print(f"  RESULT: {'SUCCESS' if ok else 'FAILED'} -- {why}", flush=True)
        for f in frames:
            writer.append_data(f)
finally:
    writer.close()

n_ok = sum(1 for ok, _, _ in results if ok)
print(f"\nwrote {args.out}")
print(f"{n_ok}/{len(results)} episodes succeeded under MPPI "
      f"({time.time() - t0:.0f} s wall clock)")
