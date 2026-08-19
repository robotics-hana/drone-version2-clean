"""Turn the lab scan (03_08_2026.glb) into MuJoCo-ready assets + lab.xml.

WHAT THE GLB ACTUALLY IS -- worth stating, because it drives everything below and
it is NOT what MuGS wants. It is a TEXTURED MESH: 101,447 vertices, 162,998 faces
in two submeshes, each with a 4K base-colour texture. A 3D Gaussian Splat is a
different thing entirely (per-Gaussian position, scale, rotation, opacity and
spherical-harmonic colour, stored as .ply), and that is what MuGS consumes. So
this gets us a mesh-rendered lab in MuJoCo; swapping in a 3DGS background later
is an independent change needing the original .ply.

Even on the MuGS path this still matters: MuGS replaces background PIXELS only.
The drone must still not fly through the walls, so the room's COLLISION geometry
has to exist in MuJoCo regardless.

THREE THINGS THIS FIXES

  1. Up-axis. glTF is Y-up, MuJoCo is Z-up. Measured, not assumed: the Y axis
     carries 95.8 m^2 of perpendicular surface (vs 54.9 and 62.5 for X and Z),
     with 31.1 m^2 facing UP at y = -1.39 (floor) and 27.3 m^2 facing DOWN at
     y = +3.0 (ceiling). A room ~4.4 m tall, so the scan is already in metres.

  2. Origin. The floor is dropped to z = 0, so world z is height above the lab
     floor -- which is what every waypoint and camera pose in SkyGrip_full.xml
     already assumes.

  3. Collision. The mesh is exported VISUAL-ONLY and the room's physics is built
     from primitive boxes. Not laziness: MuJoCo collides a mesh as its CONVEX
     HULL, and the hull of a room is a solid block, so a collidable room mesh
     spawns the drone inside solid material. Same trap as the gripper housing.

Also locates the BLUE TRAINING MAT by sampling the base-colour texture at each
floor triangle's UV. Colour alone is not enough -- see find_blue_mat.

Run:  python prepare_lab_scene.py
"""

import os

import numpy as np
import trimesh
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
GLB = os.path.join(HERE, "03_08_2026.glb")
ASSETS = os.path.join(HERE, "lab_assets")

# glTF Y-up -> MuJoCo Z-up: x->x, y->z, z->-y (a -90 deg turn about X). det = +1.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


def floor_height(mesh):
    """Height of the largest upward-facing horizontal surface, in glTF coords."""
    up = mesh.face_normals[:, 1] > 0.94
    pos, ar = mesh.triangles_center[up][:, 1], mesh.area_faces[up]
    lo, hi = mesh.bounds[0][1], mesh.bounds[1][1]
    hist, edges = np.histogram(pos, bins=60, range=(lo, hi), weights=ar)
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def largest_rect(grid):
    """Largest all-true axis-aligned rectangle (histogram sweep)."""
    best = (0, None)
    h = np.zeros(grid.shape[1], dtype=int)
    for i in range(grid.shape[0]):
        h = np.where(grid[i], h + 1, 0)
        st = []
        for j in range(grid.shape[1] + 1):
            cur = h[j] if j < grid.shape[1] else 0
            start = j
            while st and st[-1][1] >= cur:
                s0, hh = st.pop()
                area = hh * (j - s0)
                if area > best[0]:
                    best = (area, (i - hh + 1, i, s0, j - 1))
                start = s0
            st.append((start, cur))
    return best


def find_blue_mat(parts, floor_z, tol=0.30, res=0.10):
    """Locate the blue training mat and return its usable rectangle.

    Colour alone is not enough. The raw blue test matches ~65% of the floor, but
    the mat is shot through with white streaks -- HOLES IN THE SCAN, not non-blue
    floor. Taken literally they fragment the mat so badly that the largest clean
    rectangle comes out 2.70 x 1.50 m against a mat that is visibly about 4 x 6.
    So the mask is rasterised at 10 cm, closed over gaps up to 50 cm, hole-filled
    and reduced to its largest blob before measuring: 4.0 m^2 -> 24.9 m^2.
    """
    tri, col = [], []
    for _, g in parts:
        tex = getattr(g.visual.material, "baseColorTexture", None)
        if tex is None or getattr(g.visual, "uv", None) is None:
            continue
        img = np.asarray(tex.convert("RGB"), dtype=np.float32) / 255.0
        H, W = img.shape[:2]
        m = g.copy()
        m.vertices = m.vertices @ Y_UP_TO_Z_UP.T
        m.vertices[:, 2] -= floor_z
        sel = (m.face_normals[:, 2] > 0.85) & (m.triangles_center[:, 2] < tol)
        if not sel.any():
            continue
        uv = g.visual.uv[m.faces[sel]].mean(axis=1)
        px = np.clip((uv[:, 0] * (W - 1)).astype(int), 0, W - 1)
        py = np.clip(((1.0 - uv[:, 1]) * (H - 1)).astype(int), 0, H - 1)
        tri.append(m.triangles_center[sel][:, :2])
        col.append(img[py, px])
    if not tri:
        return None
    T, C = np.concatenate(tri), np.concatenate(col)
    blue = (C[:, 2] - np.maximum(C[:, 0], C[:, 1])) > 0.05

    xe = ye = np.arange(-6, 6 + res, res)
    hb, _, _ = np.histogram2d(T[blue, 0], T[blue, 1], bins=[xe, ye])
    ha, _, _ = np.histogram2d(T[:, 0], T[:, 1], bins=[xe, ye])
    grid = (hb > 0) & (hb >= 0.6 * np.maximum(ha, 1))
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, np.ones((5, 5))))
    lab, n = ndimage.label(grid)
    if n:
        grid = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    _, (i0, i1, j0, j1) = largest_rect(grid)
    rgb = C[blue].mean(axis=0) if blue.any() else np.array([0.16, 0.42, 0.74])
    # The largest ALL-BLUE rectangle is the mat: it is what gets cut and re-laid
    # flat, and it is the workspace. The blob's bounding box (bx*/by*) is also
    # returned for reference but is NOT the mat -- it overshoots on every side.
    ii, jj = np.where(grid)
    return dict(x0=float(xe[i0]), x1=float(xe[i1 + 1]),
                y0=float(ye[j0]), y1=float(ye[j1 + 1]), cells=int(grid.sum()),
                bx0=float(xe[ii.min()]), bx1=float(xe[ii.max() + 1]),
                by0=float(ye[jj.min()]), by1=float(ye[jj.max() + 1]),
                rgb=tuple(float(v) for v in rgb))


def strip_mat_faces(mesh, mat, tol=0.30, margin=0.15):
    """Clear the mat footprint of scanned geometry so one flat box can replace it.

    Cuts EVERY triangle below `tol` inside the mat's footprint, regardless of
    colour. Two earlier attempts were not enough:

      * cutting only the largest inner rectangle left the ragged blue fringe
        outside it, floating a few centimetres off the flat box -- a dark torn
        border round a clean mat, which read worse than the bumpy original;
      * cutting by colour removed the blue but left the DARK triangles that the
        colour test rejected -- shadowed mat edges, seams and scan debris. Those
        are still part of the mat and still bumpy, so the mat kept its dirty
        navy patches.

    A real crash mat is one flat rectangle, so the honest fix is to clear the
    footprint and lay a single box over it.

    The footprint is the largest ALL-BLUE rectangle, not the full blue blob. The
    blob's bounding box runs 6.30 x 6.40 m and overshoots the real mat on every
    side -- it swallows the cream floor by the table on +x and pushes past the
    stairs on -y, because the blob is a ragged shape and its bounding box is not.
    The inner rectangle's four edges land on the actual boundaries: the benches
    at x = -1.5, the table/cream floor at x = +2.8, the stairs at y = -1.9 and
    the wall at y = +3.9.

    Anything genuinely standing ON the mat would be removed too. That is fine
    here -- the mat is the clear flight area -- but it is why this is a footprint
    cut and not a global "delete everything near the floor".
    """
    if mat is None:
        return mesh, 0
    c = mesh.triangles_center
    kill = ((c[:, 2] < tol)
            & (c[:, 0] > mat["x0"]) & (c[:, 0] < mat["x1"])
            & (c[:, 1] > mat["y0"]) & (c[:, 1] < mat["y1"]))
    if kill.any():
        mesh.update_faces(~kill)
    return mesh, int(kill.sum())


def write_lab_xml(lo, hi, mat):
    """Emit lab.xml -- scan as visual-only geometry plus primitive collision."""
    pad = 0.05
    zc = float(min(hi[2], 4.40))          # ignore stray scan junk above the ceiling
    hx, hy = (hi[0] - lo[0]) / 2 + pad, (hi[1] - lo[1]) / 2 + pad
    geoms = [f'      <geom type="mesh" mesh="lab_part{i}" material="lab_mat{i}"\n'
             f'            contype="0" conaffinity="0" group="2"/>' for i in (0, 1)]
    for n, p, s in [("wall_xlo", f"{lo[0]-pad:.3f} 0 {zc/2:.3f}", f"{pad} {hy:.3f} {zc/2:.3f}"),
                    ("wall_xhi", f"{hi[0]+pad:.3f} 0 {zc/2:.3f}", f"{pad} {hy:.3f} {zc/2:.3f}"),
                    ("wall_ylo", f"0 {lo[1]-pad:.3f} {zc/2:.3f}", f"{hx:.3f} {pad} {zc/2:.3f}"),
                    ("wall_yhi", f"0 {hi[1]+pad:.3f} {zc/2:.3f}", f"{hx:.3f} {pad} {zc/2:.3f}"),
                    ("ceiling",  f"0 0 {zc+pad:.3f}", f"{hx:.3f} {hy:.3f} {pad}")]:
        geoms.append(f'      <geom name="{n}" type="box" pos="{p}" size="{s}" '
                     f'rgba="0.5 0.5 0.5 0" group="3"/>')
    note = ""
    if mat:
        cx, cy = (mat["x0"] + mat["x1"]) / 2, (mat["y0"] + mat["y1"]) / 2
        hx_m, hy_m = (mat["x1"] - mat["x0"]) / 2, (mat["y1"] - mat["y0"]) / 2
        r, g, b = mat["rgb"]
        # Flat replacement for the cut-out scan. 5 mm proud of the floor plane so
        # it wins the depth test against it; visual only, the plane is still what
        # the drone lands on. Colour is the mean of the scanned mat's own texels,
        # so it matches the surrounding scanned floor rather than being invented.
        geoms.append(f'      <geom name="blue_mat" type="box" pos="{cx:.3f} {cy:.3f} 0.005" '
                     f'size="{hx_m:.3f} {hy_m:.3f} 0.005" rgba="{r:.3f} {g:.3f} {b:.3f} 1" '
                     f'contype="0" conaffinity="0" group="2"/>')
        note = (f"\n       BLUE MAT (training area): x {mat['x0']:+.2f}..{mat['x1']:+.2f}, "
                f"y {mat['y0']:+.2f}..{mat['y1']:+.2f}, centre ({cx:+.2f}, {cy:+.2f}), "
                f"{mat['x1']-mat['x0']:.2f} x {mat['y1']-mat['y0']:.2f} m. The scanned\n"
                f"       triangles there are DELETED and replaced by the flat 'blue_mat' box:\n"
                f"       the real mat is flat, the scan of it is bumpy and full of holes.")
    xml = ('<mujocoinclude>\n'
           '  <!-- Lab scan. GENERATED by prepare_lab_scene.py -- do not hand-edit.\n'
           '       Floor is z=0, so world z is height above the lab floor.\n'
           '       The mesh geoms are VISUAL ONLY (contype/conaffinity 0): MuJoCo collides\n'
           '       a mesh as its convex hull, and the hull of a room is a solid block, which\n'
           '       would bury the drone. Physics is the invisible wall/ceiling boxes below.'
           + note + ' -->\n'
           '  <asset>\n'
           '    <texture name="lab_tex0" type="2d" file="lab_assets/lab_part0.png"/>\n'
           '    <texture name="lab_tex1" type="2d" file="lab_assets/lab_part1.png"/>\n'
           '    <material name="lab_mat0" texture="lab_tex0" specular="0.1" shininess="0.1"/>\n'
           '    <material name="lab_mat1" texture="lab_tex1" specular="0.1" shininess="0.1"/>\n'
           '    <mesh name="lab_part0" file="lab_assets/lab_part0.obj"/>\n'
           '    <mesh name="lab_part1" file="lab_assets/lab_part1.obj"/>\n'
           '  </asset>\n'
           '  <worldbody>\n'
           '    <body name="lab" pos="0 0 0">\n'
           + '\n'.join(geoms) + '\n'
           '    </body>\n'
           '  </worldbody>\n'
           '</mujocoinclude>\n')
    open(os.path.join(HERE, "lab.xml"), "w", encoding="utf-8").write(xml)
    print(f"\nwrote lab.xml (visual mesh + 5 collision primitives, ceiling {zc:.2f} m)")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    parts = list(trimesh.load(GLB).geometry.items())
    fz = floor_height(trimesh.util.concatenate([g for _, g in parts]))
    print(f"floor found at glTF y = {fz:+.3f} m -> shifted to MuJoCo z = 0")

    # The mat has to be located BEFORE export, because its triangles are cut out
    # of the exported mesh and replaced by a flat box.
    mat = find_blue_mat(parts, fz)

    lo = hi = None
    cut = 0
    for i, (_, g) in enumerate(parts):
        m = g.copy()
        m.vertices = m.vertices @ Y_UP_TO_Z_UP.T
        m.vertices[:, 2] -= fz
        lo = m.bounds[0] if lo is None else np.minimum(lo, m.bounds[0])
        hi = m.bounds[1] if hi is None else np.maximum(hi, m.bounds[1])
        m, n_cut = strip_mat_faces(m, mat)
        cut += n_cut
        m.export(os.path.join(ASSETS, f"lab_part{i}.obj"), include_texture=True)
        tex = getattr(g.visual.material, "baseColorTexture", None)
        if tex is not None:
            tex.convert("RGB").save(os.path.join(ASSETS, f"lab_part{i}.png"))
        print(f"  lab_part{i}: {len(m.faces)} faces kept ({n_cut} mat faces cut), "
              f"texture {tex.size[0]}x{tex.size[1]}")

    print(f"\nroom bounds (m): x {lo[0]:+.2f}..{hi[0]:+.2f}  y {lo[1]:+.2f}..{hi[1]:+.2f}  "
          f"z {lo[2]:+.2f}..{hi[2]:+.2f}")
    if cut:
        print(f"cut {cut} bumpy scanned mat triangles -> replaced by one flat box")

    if mat:
        cx, cy = (mat["x0"] + mat["x1"]) / 2, (mat["y0"] + mat["y1"]) / 2
        w, d = mat["x1"] - mat["x0"], mat["y1"] - mat["y0"]
        print(f"\nBLUE MAT -- the training area:")
        print(f"   usable rectangle x {mat['x0']:+.2f}..{mat['x1']:+.2f}  "
              f"y {mat['y0']:+.2f}..{mat['y1']:+.2f}")
        print(f"   centre ({cx:+.2f}, {cy:+.2f})   {w:.2f} x {d:.2f} m = {w*d:.1f} m^2")
    else:
        print("\nNo blue mat found -- widen the colour test.")
    write_lab_xml(lo, hi, mat)


if __name__ == "__main__":
    main()
