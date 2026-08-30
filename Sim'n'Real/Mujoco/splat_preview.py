"""Quick-look preview of a 3D Gaussian-splat capture (.ply), rendered as
an orbiting point cloud. NOT a MuJoCo tool and not a real splat
rasteriser: every Gaussian is drawn as a small opaque dot, so flat
surfaces show speckle where the true splats would blend. It exists to
sanity-check a capture's geometry, colour and coverage before deciding
whether to register it into the sim -- for photoreal views use a proper
3DGS viewer (SuperSplat, gsplat, the capture tool's own renderer).

Reads the standard 3DGS property layout written by most trainers:
    x y z  nx ny nz  f_dc_0..2  f_rest_0..44  opacity  scale_0..2  rot_0..3
i.e. 62 float32 per vertex. Colour comes from the DC spherical-harmonic
band (f_dc), opacity through a sigmoid.

usage:
    python splat_preview.py capture.ply out.mp4 [--frames 72] [--keep 0.70]
"""
import argparse

import numpy as np
import imageio.v2 as imageio

SH_C0 = 0.28209479177387814      # Y(0,0): DC band -> base colour
N_PROPS = 62


def load(path):
    """Return (xyz, rgb uint8) for the Gaussians worth drawing."""
    with open(path, "rb") as fh:
        head = b""
        while b"end_header" not in head:
            chunk = fh.read(1024)
            if not chunk:
                raise ValueError("no PLY header terminator")
            head += chunk
    offset = head.index(b"end_header") + len(b"end_header") + 1
    raw = np.fromfile(path, dtype=np.float32, offset=offset)
    n = raw.size // N_PROPS
    raw = raw[:n * N_PROPS].reshape(n, N_PROPS)
    xyz = raw[:, 0:3]
    rgb = np.clip(0.5 + SH_C0 * raw[:, 6:9], 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-raw[:, 54]))
    solid = opacity > 0.35          # the rest is haze the viewer can't show
    return xyz[solid], (rgb[solid] * 255).astype(np.uint8)


def core(xyz, rgb, keep):
    """Drop the floater shell most captures wrap around the real scene."""
    r = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
    m = r < np.percentile(r, keep * 100.0)
    return xyz[m], rgb[m]


def orbit(xyz, rgb, n_frames, size=640, fov_deg=33.0):
    """Yield frames of a circular orbit about the scene's vertical axis."""
    c = np.median(xyz, axis=0)
    lo, hi = np.percentile(xyz, 3, axis=0), np.percentile(xyz, 97, axis=0)
    ext = hi - lo
    # the thinnest extent is the room's height: floor-to-ceiling is the
    # short dimension of a wide capture
    up_ax = int(np.argmin(ext))
    below = (xyz[:, up_ax] < np.median(xyz[:, up_ax])).sum()
    up = np.zeros(3)
    up[up_ax] = 1.0 if below > len(xyz) / 2 else -1.0
    ax = [i for i in range(3) if i != up_ax]
    radius = 0.62 * max(ext[ax[0]], ext[ax[1]])
    f = size / (2.0 * np.tan(np.radians(fov_deg)))
    for k in range(n_frames):
        th = 2.0 * np.pi * k / n_frames
        cam = c.copy()
        cam[ax[0]] += 1.35 * radius * np.cos(th)
        cam[ax[1]] += 1.35 * radius * np.sin(th)
        cam += up * 0.72 * radius
        fwd = (c + up * 0.10 * radius) - cam
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        p = xyz - cam
        z = p @ fwd
        front = z > 0.1
        u = (p[front] @ right) / z[front] * f + size / 2
        v = (p[front] @ down) / z[front] * f + size / 2
        inb = (u >= 1) & (u < size - 2) & (v >= 1) & (v < size - 2)
        u, v = u[inb].astype(int), v[inb].astype(int)
        # painter's algorithm: far points first, near ones overwrite
        order = np.argsort(-z[front][inb])
        u, v, col = u[order], v[order], rgb[front][inb][order]
        img = np.zeros((size, size, 3), np.uint8)
        for du in range(3):
            for dv in range(3):
                img[v + dv, u + du] = col
        yield img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--keep", type=float, default=0.70,
                    help="radial fraction kept; lower prunes more floaters")
    a = ap.parse_args()
    xyz, rgb = load(a.ply)
    print("solid gaussians: %d" % len(xyz))
    xyz, rgb = core(xyz, rgb, a.keep)
    print("after floater prune: %d" % len(xyz))
    w = imageio.get_writer(a.out, fps=12, codec="libx264", quality=8,
                           macro_block_size=1)
    for img in orbit(xyz, rgb, a.frames):
        w.append_data(img)
    w.close()
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
