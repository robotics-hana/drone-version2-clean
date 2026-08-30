"""Three-view review video of the SkyGrip pick-and-place experiment
(collect_demos.py), run against vla_drone.xml -- the editable flattened copy
of SkyGrip_full.xml (wrist_cam re-aimed dead-centre on grasp_site, scene_cam
pitched 45 deg to the table). The original model files stay untouched.

The onboard wrist_cam and scene_cam render at their exact XML poses: the
per-episode camera-pose jitter collect_demos applies for domain
randomisation is pinned back to nominal here, render-side only, so the views
do not wander between episodes. The third view is a FRONT camera added in
code (not in any XML): a free camera on the -y side of the workspace -- the
side the XML's own overview/external cameras already view from -- looking
head-on along +y at the table, so the drone, the table and all three
objects are seen straight-on while it picks and places.

Layout per frame (960x480):
    +----------------------+-----------+
    |                      | wrist_cam |
    |      FRONT view      +-----------+
    |      (640x480)       | scene_cam |
    +----------------------+-----------+

Follows render_trials.py's viewing-run recipe: picks only, failed attempts
discarded, so every episode in the video is a success.
"""
import argparse, io, contextlib
import numpy as np
import mujoco
import imageio.v2 as imageio
import collect_demos as C

# Labels are a nicety, not a dependency: no PIL, no labels.
# Font is looked up across platforms so a cluster (Linux) render gets the same
# legible banner as a local (Windows) one, rather than the tiny bitmap default.
try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT = None
    for _path in ("C:/Windows/Fonts/arialbd.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            _FONT = ImageFont.truetype(_path, 18)
            break
        except OSError:
            continue
    if _FONT is None:
        _FONT = ImageFont.load_default()
except ImportError:
    Image = None

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--out", type=str, default="../../Reports/pick_run_5ep_3view.mp4")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--distance", type=float, default=1.35,
                help="front-camera distance from the table, m")
args = ap.parse_args()

# Same viewing-run settings render_trials.py uses: picks only, an even
# forward-reach/overhead mix, and a DESCEND budget long enough not to starve
# the servo phase.
C.EPISODE_MODE_P = [1.0, 0.0, 0.0]
C.REACH_FRACTION = 0.5
C.PHASE_FRAME_BUDGET["DESCEND"] = int(8.0 * C.FPS)

with contextlib.redirect_stdout(io.StringIO()):
    ctrl = C.SkyGripController("vla_drone.xml", C.MPPIParams(50, 16),
                               flight="pd")
model, data = ctrl.model, ctrl.data
ik = C.ArmIK(model)
rend_front = mujoco.Renderer(model, height=480, width=640)
rend_small = mujoco.Renderer(model, height=240, width=320)
rng = np.random.default_rng(args.seed)

# Nominal (XML) poses of the onboard cameras, captured before any episode has
# jittered them. grab() writes these back every frame, so the rendered views
# sit exactly where the XML puts them; the jitter still applies to what the
# EXPERIMENT records, this is render-side only.
_PINNED = {}
for _name in ("wrist_cam", "scene_cam"):
    _cid = model.camera(_name).id
    _PINNED[_cid] = (model.cam_pos[_cid].copy(), model.cam_quat[_cid].copy())


def pin_cams():
    for _cid, (p, q) in _PINNED.items():
        model.cam_pos[_cid] = p
        model.cam_quat[_cid] = q

ped = model.body("pedestal").id
ped_gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ped][0]

front = mujoco.MjvCamera()
front.azimuth, front.elevation, front.distance = 90, -14, args.distance


def front_lookat():
    # Head-on at the table. The pedestal pose is randomised per episode but
    # static within one, so this is a FIXED camera for the whole episode --
    # re-read each frame only because it is cheap and always current.
    surface_z = model.body_pos[ped][2] + model.geom_size[ped_gid, 2]
    return (model.body_pos[ped][0], model.body_pos[ped][1], surface_z + 0.15)


def annotate(img, banner):
    if Image is None:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    for text, (x, y) in ((banner, (8, 8)),
                         ("wrist_cam", (648, 8)),
                         ("scene_cam", (648, 248))):
        w = dr.textlength(text, font=_FONT)
        dr.rectangle([x, y, x + w + 12, y + 26], fill=(0, 0, 0, 150))
        dr.text((x + 6, y + 4), text, font=_FONT, fill=(255, 255, 255))
    return np.asarray(im)


writer = imageio.get_writer(args.out, fps=C.FPS, codec="libx264",
                            quality=8, macro_block_size=1)
banked = attempts = 0
try:
    while banked < args.episodes and attempts < args.episodes * 4:
        attempts += 1
        frames = []

        def grab():
            pin_cams()
            front.lookat[:] = front_lookat()
            rend_front.update_scene(data, camera=front)
            tile_front = rend_front.render().copy()
            rend_small.update_scene(data, camera="wrist_cam")
            tile_wrist = rend_small.render().copy()
            rend_small.update_scene(data, camera="scene_cam")
            tile_scene = rend_small.render().copy()
            frames.append(np.hstack([tile_front,
                                     np.vstack([tile_wrist, tile_scene])]))

        with contextlib.redirect_stdout(io.StringIO()):
            fr, info = C.run_episode(model, data, None, ctrl, ik, rng,
                                     "Pick up the {colour} {shape} and place it",
                                     on_frame=grab)
            ok, why = C.episode_succeeded(model, data, info)
        if not ok:
            print(f"  attempt {attempts}: discarded ({why})", flush=True)
            continue
        banked += 1
        banner = (f"episode {banked}/{args.episodes} - pick up the "
                  f"{info['target_colour']} {info['target_shape']}")
        style = "forward-reach" if info["reach_y"] is not None else "overhead"
        print(f"episode {banked}/{args.episodes}: {info['target_colour']} "
              f"{info['target_shape']:8s} | {style:13s} | {len(frames):4d} frames | {why}",
              flush=True)
        for f in frames:
            writer.append_data(annotate(f, banner))
finally:
    writer.close()
print(f"\nwrote {args.out}  ({banked} episodes, {attempts} attempts)")
