"""discrimination_render.py -- Test B: can the penguin and the weight be
told apart at the policy's input resolution?

The oracle-vector result cleared perception for LOCALISATION but said
nothing about DISCRIMINATION -- handing over the target vector bypasses
selection entirely. If the two candidates are indistinguishable at 224 px
from the distances where the policy commits, no amount of language
training can bind the noun to the right object.

Method: a trained-geometry scene (both objects side by side on the table,
0.50 m apart -- the trained separation is 0.45-0.62 m) rendered through
the three policy cameras at 224 px with the drone at the three distance
bands that matter. Per stream x band x object: segmentation pixel count,
mean RGB over the object's pixels, and within-object RGB std; separation
is judged by the mean-colour distance against the pooled std, plus pixel
area. A crop montage is saved for the sign-off package.

Decision rule (pre-registered, from the brief, before any numbers):
  if the objects are NOT separable in any stream at 1.0-1.5 m, the
  camera3 re-frame is justified -- but for DISCRIMINATION, not
  localisation, which is a different claim from D47/D49 and must be
  recorded as such.

usage: python discrimination_render.py <out_montage.png>
(run from Sim'n'Real/Mujoco so collect_airvla imports)
"""
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

import collect_airvla as A

RES = 224
BANDS = [("1.0-1.5 m", 1.25, 0.60), ("0.6-1.0 m", 0.80, 0.60),
         ("0.4-0.6 m", 0.50, 0.55)]        # (label, drone dist, altitude)
CAMS = [("lab_external", "camera3"), ("scene_cam", "camera2 nose"),
        ("wrist_cam", "camera1 wrist")]

out = sys.argv[1]
r = A.Runner(12345)
m, d = r.model, r.data
tgt_xy = np.array([0.0, 0.60])
r.reset_scene(tgt_xy, np.array([0.0, 1.6, 0.25]), obj="weight")
# pin the distractor at 0.50 m for a representative trained separation
pen = r.objs["plush penguin"]
d.qpos[pen["adr"]:pen["adr"] + 2] = tgt_xy + [0.50, 0.0]

rend = mujoco.Renderer(m, height=RES, width=RES)
seg = mujoco.Renderer(m, height=RES, width=RES)
seg.enable_segmentation_rendering()


def subtree(body):
    b = m.body(body).id
    return set(g for g in range(m.ngeom)
               if m.body_rootid[m.geom_bodyid[g]] == b)


G = {"weight": subtree("pick_weight"), "penguin": subtree("penguin")}
print("DISCRIMINATION  scene: weight at (0.00, 0.60), penguin at "
      "(0.50, 0.60), trained separation")
print("decision rule (pre-registered): not separable in ANY stream at "
      "1.0-1.5 m => camera3 re-frame justified for DISCRIMINATION (a "
      "different claim from the earlier localisation one)")

tiles = []
for label, dist, alt in BANDS:
    d.qpos[0:3] = [0.25, 0.60 + dist, alt]     # centred between the pair
    d.qpos[3:7] = [1, 0, 0, 0]                 # yaw 0 faces -y (the pair)
    mujoco.mj_forward(m, d)
    for cam, cname in CAMS:
        rend.update_scene(d, camera=cam)
        rgb = rend.render().copy()
        seg.update_scene(d, camera=cam)
        sg = seg.render()
        vis = sg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
        stats = {}
        for k, gs in G.items():
            mask = np.isin(sg[:, :, 0], list(gs)) & vis
            n = int(mask.sum())
            stats[k] = (n, rgb[mask].mean(0) if n else np.zeros(3),
                        rgb[mask].std(0).mean() if n else 0.0)
        nw, cw, sw = stats["weight"]
        npg, cp, sp = stats["penguin"]
        dc = float(np.linalg.norm(cw - cp)) if nw and npg else 0.0
        pooled = max(1.0, (sw + sp) / 2)
        verdict = ("separable (colour)" if nw >= 4 and npg >= 4
                   and dc > 2 * pooled else
                   "separable (area %dx)" % round(max(nw, npg)
                                                  / max(1, min(nw, npg)))
                   if nw >= 4 and npg >= 4 and (max(nw, npg)
                                                >= 3 * min(nw, npg)) else
                   "NOT separable" if (nw < 4 or npg < 4) else "marginal")
        print("%-10s %-16s weight %4dpx rgb(%3.0f,%3.0f,%3.0f)  penguin "
              "%4dpx rgb(%3.0f,%3.0f,%3.0f)  dColour %5.1f (pooled std "
              "%4.1f)  -> %s"
              % (label, cname, nw, *cw, npg, *cp, dc, pooled, verdict))
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        dr.text((4, 4), "%s | %s" % (cname, label), fill=(255, 255, 0))
        dr.text((4, RES - 14), "w:%dpx p:%dpx" % (nw, npg),
                fill=(0, 255, 255))
        tiles.append(np.asarray(im))

sheet = np.concatenate([np.concatenate(tiles[i * 3:i * 3 + 3], axis=1)
                        for i in range(len(BANDS))], axis=0)
imageio.imwrite(out, sheet)
print("montage ->", out)
