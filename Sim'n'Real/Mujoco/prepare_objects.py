"""Fetch YCB objects and work out which ones this gripper can actually grasp.

The point of measuring rather than eyeballing: this gripper opens 32 mm, full
stop -- that is the Pololu kit's range, not a sim choice. Most of the YCB set is
far too wide for it (mustard bottle ~85 mm, soup can ~66 mm, cracker box ~60 mm),
so "use YCB like the paper does" cannot be taken at face value. The paper's
UMI-derived gripper is much larger than this one.

So for each candidate this computes a GRASP PROFILE: sweeping horizontal slices
up the object, how wide is it along the jaws' closing axis? An object is
graspable if some slice is narrow enough, and USEFULLY graspable only if that
slice is near the top -- because of the second constraint, which is easy to
forget: the gripper body obstructs about 10 mm above the jaw midpoint, so the
object may only protrude ~7 mm above the grasp point. Tall things must therefore
be caught near their top, which is exactly the "bottle by the neck" case.

Two sets come out of this:
  EASY  -- fits the jaws with real clearance (markers, small tools)
  NECK  -- only graspable at a narrow neck, tight tolerance, looks like the paper

Run:  python prepare_objects.py            # measure only, no XML written
      python prepare_objects.py --write    # also emit objects.xml
"""

import os
import sys
import tarfile
import urllib.request

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "ycb_assets")
BASE = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{}_google_16k.tgz"

# Jaw geometry, from SkyGrip_full.xml. Both numbers matter.
JAW_OPEN_MM = 32.0          # act_gripper ctrlrange 0..0.016, gap = 2000 * ctrl
CLEARANCE_MM = 3.0          # wanted each side on approach
CLEAR_COLUMN_MM = 10.0      # headroom above grasp_site before the body fouls
PAD_HALF_MM = 9.0           # pad_1/pad_2 are 18 mm tall, centred on grasp_site
MIN_GRIP_MM = 6.0           # thinner than this and there is nothing to hold on to

CANDIDATES = [
    # thin/tall things -- the natural fit for a 32 mm jaw
    "040_large_marker", "037_scissors", "030_fork", "031_spoon", "032_knife",
    "033_spatula", "042_adjustable_wrench", "043_phillips_screwdriver",
    "044_flat_screwdriver", "038_padlock", "026_sponge", "011_banana",
    # small blocks
    "070-a_colored_wood_blocks", "073-a_lego_duplo", "073-b_lego_duplo",
    "065-a_cups", "065-b_cups",
    # bottles -- only graspable by a neck/cap, if at all
    "006_mustard_bottle", "021_bleach_cleanser", "022_windex_bottle",
    "003_cracker_box", "005_tomato_soup_can",
]


def fetch(name):
    """Download and extract one YCB object; returns the textured mesh path."""
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, name)
    obj = os.path.join(out, "google_16k", "textured.obj")
    if os.path.exists(obj):
        return obj
    tgz = os.path.join(ASSETS, name + ".tgz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(BASE.format(name), tgz)
    with tarfile.open(tgz) as t:
        t.extractall(ASSETS)
    os.remove(tgz)
    return obj if os.path.exists(obj) else None


def grasp_profile(mesh, n=60):
    """Width along each horizontal axis, as a function of height.

    Returns (heights, width_x, width_y) in mm, measured from the object's base.
    The object is first laid on its base -- YCB meshes are centred on their own
    origin, not resting on z=0, so a raw bounding box says nothing about what a
    gripper closing at a given height would meet.
    """
    m = mesh.copy()
    m.vertices -= m.bounds[0]                     # base to z = 0
    zmax = m.bounds[1][2]
    zs = np.linspace(0.02 * zmax, 0.98 * zmax, n)
    wx, wy = [], []
    for z in zs:
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            wx.append(np.nan)
            wy.append(np.nan)
            continue
        p = sec.vertices
        wx.append(np.ptp(p[:, 0]) * 1000.0)
        wy.append(np.ptp(p[:, 1]) * 1000.0)
    return zs * 1000.0, np.array(wx), np.array(wy)


ORIENTATIONS = {
    "as-scanned": np.eye(3),
    "upright-x": trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3],
    "upright-y": trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3],
}


def classify(name, mesh):
    """Can this gripper take it, in which pose, and at what height?

    Two hard constraints, and the second is the one that is easy to miss:

      WIDTH  -- some horizontal slice must be <= JAW_OPEN - 2*CLEARANCE.
      REACH  -- that slice must lie within CLEAR_COLUMN_MM of the object's TOP.
                The gripper body sits ~10 mm above the jaw midpoint, so anything
                protruding higher than that fouls the body on the way down. This
                is why "the object is narrow somewhere" is not enough: a spray
                bottle with a 15 mm neck 57 mm below its top is unreachable.

    Orientation is searched too, because YCB meshes are in their natural resting
    pose and several of these (markers, wrenches) are lying FLAT. Standing a
    marker up turns it from a cylinder that must be pinched across its rounded
    top -- a slipping grasp -- into a 18 mm shaft with 100 mm of parallel sides.
    """
    limit = JAW_OPEN_MM - 2 * CLEARANCE_MM
    best = None
    for pose, R in ORIENTATIONS.items():
        m = mesh.copy()
        m.vertices = m.vertices @ R.T
        h, wx, wy = grasp_profile(m)
        top = h[-1]
        w = np.fmin(wx, wy)          # jaws close on ONE axis; yaw picks the narrower
        gz = gw = None
        for i, z in enumerate(h):
            if top - z > CLEAR_COLUMN_MM:
                continue                        # body would foul what sticks up
            # The jaws are PAD_H tall and must close around everything from the
            # bottom of the pads up to the object's top, so the binding number is
            # the WIDEST slice in that band -- not the narrowest. Taking the
            # narrowest is what made a mustard bottle look like an 11 mm grasp:
            # the slice a millimetre below the tip is nearly tangent to the
            # surface, so its width tends to zero and means nothing.
            band = (h >= z - PAD_HALF_MM) & (h <= top)
            wmax = float(np.nanmax(w[band])) if band.any() else np.inf
            wmin = float(np.nanmin(w[band])) if band.any() else 0.0
            if wmax <= limit and wmin >= MIN_GRIP_MM and (gw is None or wmax < gw):
                gz, gw = float(z), wmax
        cand = dict(pose=pose, height_mm=float(top),
                    min_width_anywhere_mm=float(np.nanmin(w)),
                    graspable=gz is not None)
        if gz is not None:
            cand["grasp_z_mm"], cand["grasp_w_mm"] = gz, gw
        key = (cand["graspable"], -cand.get("grasp_w_mm", 1e9))
        if best is None or key > (best["graspable"], -best.get("grasp_w_mm", 1e9)):
            best = cand
    best["name"] = name
    return best


def main():
    print(f"jaws open {JAW_OPEN_MM:.0f} mm; allowing {CLEARANCE_MM:.0f} mm a side "
          f"-> max presented width {JAW_OPEN_MM - 2*CLEARANCE_MM:.0f} mm")
    print(f"body blocks ~{CLEAR_COLUMN_MM:.0f} mm above the grasp point, so the grasp "
          f"must be near the object's top\n")
    print(f"{'object':22s} {'pose':11s} {'height':>7s} {'narrowest':>10s}  verdict")
    rows = []
    for name in CANDIDATES:
        try:
            path = fetch(name)
            if path is None:
                print(f"{name:24s}   -- no textured mesh in archive")
                continue
            mesh = trimesh.load(path, force="mesh")
            r = classify(name, mesh)
            rows.append(r)
            if r["graspable"]:
                v = (f"GRASPABLE -- jaws span {r['grasp_w_mm']:.1f} mm at z={r['grasp_z_mm']:.0f} mm "
                     f"({(JAW_OPEN_MM-r['grasp_w_mm'])/2:.1f} mm a side)")
            elif r["min_width_anywhere_mm"] <= JAW_OPEN_MM - 2 * CLEARANCE_MM:
                v = (f"UNREACHABLE -- narrow ({r['min_width_anywhere_mm']:.1f} mm) but not "
                     f"within {CLEAR_COLUMN_MM:.0f} mm of the top")
            else:
                v = f"TOO WIDE ({r['min_width_anywhere_mm']:.1f} mm at its narrowest)"
            print(f"{name:22s} {r['pose']:11s} {r['height_mm']:6.0f}mm "
                  f"{r['min_width_anywhere_mm']:9.1f}mm  {v}")
        except Exception as e:
            print(f"{name:24s}   FAILED: {type(e).__name__}: {e}")
    return rows


if __name__ == "__main__":
    main()
