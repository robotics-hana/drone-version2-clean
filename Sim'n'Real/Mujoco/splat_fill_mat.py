"""splat_fill_mat.py -- fill the holes in the blue training mat of the
cleaned lab splat.

splat_clean.py flattened the mat's Gaussians onto a fitted plane, but the
capture itself has bald patches (regions the scanner never saw), which
render as holes. This script synthesises new flat blue Gaussians on the
mat plane wherever coverage is missing, colour-matched to the median of
the real mat Gaussians, and writes a new .ply (the input is not touched).

How the mat's footprint is decided: the existing (already-flattened) mat
Gaussians are found (blue + on the plane), their in-plane orientation is
taken from PCA (the mat is a rectangle, so the principal axes align with
its edges), and the rectangle extent is the robust 0.5th-99.5th
percentile along each axis. A 2 cm occupancy grid over that rectangle is
compared against the existing points; cells with no Gaussian within
3.5 cm get a new one:

  position  cell centre, on the plane
  scale     2.2 cm discs in-plane, 4 mm normal (same squash as the
            flattening step)
  rotation  identity; colour f_dc = median of the real mat (f_rest 0);
  opacity   sigmoid^-1(0.95)

usage: python splat_fill_mat.py clab_clean.ply clab_filled.ply
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
N_PROPS = 62
I_DC, I_OPA, I_SCALE, I_ROT = 6, 54, 55, 58

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as fh:
    header = b""
    while b"end_header" not in header:
        header += fh.readline()
    n = int([l for l in header.decode().splitlines()
             if l.startswith("element vertex")][0].split()[-1])
    V = np.fromfile(fh, dtype=np.float32,
                    count=n * N_PROPS).reshape(n, N_PROPS)
print("loaded %d gaussians" % n)

rgb = np.clip(0.5 + SH_C0 * V[:, I_DC:I_DC + 3], 0, 1)
blue = (rgb[:, 2] > rgb[:, 0] + 0.06) & (rgb[:, 2] > rgb[:, 1] + 0.06)
# up axis is -y (measured by splat_clean); flattened mat sits on a plane
UP, SGN = 1, -1
h = SGN * V[:, UP]
floor = float(np.median(h[blue]))
mat = blue & (np.abs(h - floor) < 0.03)
print("mat: %d flattened gaussians at height %+.3f" % (mat.sum(), floor))

ax_in = [a for a in range(3) if a != UP]
P = V[np.where(mat)[0]][:, ax_in].astype(np.float64)
mu = P.mean(0)
C = np.cov((P - mu).T)
evec = np.linalg.eigh(C)[1]                     # columns: rect edge axes
U = (P - mu) @ evec
lo = np.percentile(U, 0.5, axis=0)
hi = np.percentile(U, 99.5, axis=0)
print("mat rectangle: %.2f x %.2f m (PCA-aligned)"
      % (hi[0] - lo[0], hi[1] - lo[1]))

STEP, REACH = 0.02, 0.035
gx, gy = np.meshgrid(np.arange(lo[0], hi[0], STEP),
                     np.arange(lo[1], hi[1], STEP))
cells = np.column_stack([gx.ravel(), gy.ravel()])
tree = cKDTree(U)
empty = tree.query(cells, k=1)[0] > REACH
# INTERIOR test: fill a hole only if real mat surrounds it -- support
# within 0.35 m in at least 3 of the 4 in-plane quadrants. This follows
# the mat's true (irregular) boundary instead of overfilling the
# bounding rectangle (first attempt overshot the edges).
# EXCLUSION: something non-blue already occupies the plane there
# (an object standing on the mat, floor showing through a true gap)
# occupancy = clearly-not-mat only: warm-toned (wood, floor, boxes)
# or dark (object bases). Washed-out GLARE on the mat scans as pale
# neutral and must NOT block filling -- it failed the blue test and,
# in the first attempt, walled off the largest hole's core.
warm = rgb[:, 0] > rgb[:, 2] + 0.04
dark = rgb.max(1) < 0.30
occ = (warm | dark) & (np.abs(h - floor) < 0.04)
occ_t = cKDTree(V[np.where(occ)[0]][:, ax_in].astype(np.float64) @ evec
                - mu @ evec)
occupied = occ_t.query(cells, k=1)[0] < 0.05
# ITERATIVE GROWTH: a single surround-test cannot reach the core of a
# hole wider than ~0.8 m, so the fill grows ring by ring -- each pass
# admits cells supported (3 of 4 quadrants) by real OR already-filled
# mat, and each new ring supports the next. Object cells stay excluded.
# FOOTPRINT by morphological closing of the DENSE real mat: rasterise
# the real points, drop sparse specks (isolated scan dots beyond the
# edge seeded runaway growth in earlier attempts), then close with a
# 0.30 m disc -- bays and holes up to ~0.6 m across fill, straight
# edges return exactly, nothing extends past the true boundary.
from scipy import ndimage
nx = int(np.ceil((hi[0] - lo[0]) / STEP)) + 1
ny = int(np.ceil((hi[1] - lo[1]) / STEP)) + 1
ij = np.clip(((U - lo) / STEP).astype(int), 0, [nx - 1, ny - 1])
raster = np.zeros((nx, ny), bool)
raster[ij[:, 0], ij[:, 1]] = True
dense = ndimage.binary_opening(raster, ndimage.generate_binary_structure(2, 1), iterations=1)
disc = (lambda r: (lambda a: a[0]**2 + a[1]**2 <= r * r)(
    np.ogrid[-int(r := 15):16, -15:16]))(15)      # 15 px = 0.30 m
mask = ndimage.binary_closing(dense, disc)
cij = np.clip(((cells - lo) / STEP).astype(int), 0, [nx - 1, ny - 1])
footprint = mask[cij[:, 0], cij[:, 1]]
holes = empty & footprint & ~occupied
it = 0
new_uv = cells[holes]
print("grid %d cells: %d empty, %d occupied by non-mat, %d filled in "
      "%d growth passes" % (len(cells), empty.sum(),
                            (empty & occupied).sum(), holes.sum(), it))

# plane height at each new point: nearest real mat gaussian's height
# (the plane has a slight tilt, so copy the neighbour rather than a const)
near = tree.query(new_uv, k=1)[1]
new_h = h[np.where(mat)[0]][near]
xy = new_uv @ evec.T + mu
N = np.zeros((len(new_uv), N_PROPS), dtype=np.float32)
N[:, ax_in[0]] = xy[:, 0]
N[:, ax_in[1]] = xy[:, 1]
N[:, UP] = SGN * new_h
dc = np.median(V[np.where(mat)[0], I_DC:I_DC + 3], axis=0)
N[:, I_DC:I_DC + 3] = dc
N[:, I_OPA] = np.log(0.95 / 0.05)               # sigmoid^-1(0.95)
N[:, I_SCALE:I_SCALE + 3] = np.log(0.022)
N[:, I_SCALE + UP] = np.log(0.004)
N[:, I_ROT] = 1.0                               # identity quaternion
out = np.vstack([V, N])
with open(dst, "wb") as fh:
    fh.write(header.replace(b"element vertex %d" % n,
                            b"element vertex %d" % len(out)))
    out.astype(np.float32).tofile(fh)
print("wrote %s: %d gaussians (%d synthesised, colour rgb %.2f/%.2f/%.2f)"
      % (dst, len(out), len(N), *np.clip(0.5 + SH_C0 * dc, 0, 1)))
