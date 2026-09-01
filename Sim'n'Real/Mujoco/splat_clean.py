"""splat_clean.py -- clean the lab's 3D Gaussian-splat capture: remove
floating blobs and flatten the blue training mat.

Operates on the raw 3DGS .ply (62 float32 properties per Gaussian, the
standard trainer layout that splat_preview.py documents) and writes a new
.ply with the same layout, so any splat viewer or the future MuGS
background path consumes it unchanged. The source file is never modified.

What it does, in order, each step reported with counts:

  1. FRAME DETECTION. The capture's up-axis and floor height are measured,
     not assumed: for each axis, the fraction of Gaussians concentrated in
     a thin slab near the axis extreme is computed; the floor is the
     dominant slab. (The GLB the sim mesh came from was Y-up; this capture
     is detected independently.)
  2. FLOATER REMOVAL. Statistical outlier rejection: distance to the k-th
     nearest neighbour (k=8), cut at mean + 3 sigma. Isolated low-opacity
     Gaussians (sigmoid opacity < 0.10 and k-NN distance beyond mean +
     1 sigma) go too -- these are the classic hazy blobs.
  3. BELOW-FLOOR REMOVAL. Gaussians more than 6 cm below the detected
     floor plane are scan noise.
  4. MAT FLATTENING. Blue Gaussians (DC colour with b > r+0.06 and
     b > g+0.06) within 12 cm of the floor plane define the mat; a plane
     is refitted to them by least squares, then every mat Gaussian is
     projected onto that plane, its normal-direction scale squashed so it
     renders as a flat disc. Blue Gaussians hovering 12-40 cm above the
     plane are mat ghosts and are deleted.

usage:
    python splat_clean.py "clab (4).ply" clab_clean.ply
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
N_PROPS = 62
# property indices in the standard layout
IX, IY, IZ = 0, 1, 2
I_DC = 6                       # f_dc_0..2
I_OPA = 54                     # opacity (pre-sigmoid)
I_SCALE = 55                   # scale_0..2 (log)

src, dst = sys.argv[1], sys.argv[2]

with open(src, "rb") as fh:
    header = b""
    while b"end_header" not in header:
        header += fh.readline()
    n = int([l for l in header.decode().splitlines()
             if l.startswith("element vertex")][0].split()[-1])
    assert header.count(b"property float") == N_PROPS, "unexpected layout"
    V = np.fromfile(fh, dtype=np.float32, count=n * N_PROPS)
V = V.reshape(n, N_PROPS).copy()
print("loaded %d gaussians from %s" % (n, src))

xyz = V[:, 0:3]
opa = 1.0 / (1.0 + np.exp(-V[:, I_OPA]))
rgb = np.clip(0.5 + SH_C0 * V[:, I_DC:I_DC + 3], 0, 1)

# -- 1. frame detection ------------------------------------------------
best = None
for ax in range(3):
    for sign in (+1, -1):
        v = sign * xyz[:, ax]
        lo = np.percentile(v, 2)
        frac = float(np.mean(np.abs(v - lo) < 0.05))
        if best is None or frac > best[0]:
            best = (frac, ax, sign, lo)
frac, UP, SGN, floor_raw = best
h = SGN * xyz[:, UP]                       # height above raw floor, signed
floor = float(np.median(h[np.abs(h - floor_raw) < 0.05]))
print("frame: up-axis %s%s, floor at %+.3f (%.1f%% of gaussians in the "
      "floor slab)" % ("+" if SGN > 0 else "-", "xyz"[UP], floor,
                       100 * frac))

# -- 2. floaters -------------------------------------------------------
tree = cKDTree(xyz)
dk = tree.query(xyz, k=9)[0][:, -1]        # distance to 8th neighbour
cut = dk.mean() + 3 * dk.std()
sor = dk > cut
hazy = (opa < 0.10) & (dk > dk.mean() + dk.std())
drop = sor | hazy
print("floaters: %d isolated (kNN > %.3f m) + %d hazy low-opacity = "
      "%d removed" % (sor.sum(), cut, (hazy & ~sor).sum(), drop.sum()))

# -- 3. below floor ----------------------------------------------------
below = (h < floor - 0.06) & ~drop
drop |= below
print("below-floor: %d removed" % below.sum())

# -- 4. blue mat -------------------------------------------------------
blue = (rgb[:, 2] > rgb[:, 0] + 0.06) & (rgb[:, 2] > rgb[:, 1] + 0.06)
mat = blue & (np.abs(h - floor) < 0.12) & ~drop
ghosts = blue & (h - floor >= 0.12) & (h - floor < 0.40) & ~drop
if mat.sum() > 1000:
    # least-squares plane over the mat points, in the two in-plane axes
    ax_in = [a for a in range(3) if a != UP]
    A = np.column_stack([xyz[mat][:, ax_in], np.ones(mat.sum())])
    b = h[mat]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    tilt = np.degrees(np.arctan(np.linalg.norm(coef[:2])))
    plane_h = A @ coef
    resid = b - plane_h
    print("mat: %d blue gaussians near floor; fitted plane tilt %.2f deg, "
          "height residual rms %.1f mm (max %.1f mm) -- flattening"
          % (mat.sum(), tilt, 1000 * np.sqrt(np.mean(resid ** 2)),
             1000 * np.abs(resid).max()))
    V[np.where(mat)[0], UP] = (SGN * plane_h).astype(np.float32)
    V[np.where(mat)[0], I_SCALE + UP] = np.float32(np.log(0.004))
    drop |= ghosts
    print("mat ghosts (blue, 12-40 cm up): %d removed" % ghosts.sum())
else:
    print("WARNING: no blue mat found near the floor -- nothing flattened")

keep = ~drop
out = V[keep]
with open(dst, "wb") as fh:
    fh.write(header.replace(b"element vertex %d" % n,
                            b"element vertex %d" % keep.sum()))
    out.astype(np.float32).tofile(fh)
print("kept %d / %d gaussians (%.1f%%) -> %s"
      % (keep.sum(), n, 100 * keep.mean(), dst))
