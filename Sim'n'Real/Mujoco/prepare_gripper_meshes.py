"""Bake the new (Pololu Micro Gripper Kit) meshes into MuJoCo-ready STLs.

Why this script exists at all
----------------------------
The three new gripper parts arrived as slicer files -- `gripper-base.3mf`,
`left clamp.3mf` and `right clamp.stl`. Two problems with using them directly:

  1. MuJoCo cannot read 3MF. Only STL / OBJ / MSH.
  2. Slicer files carry BUILD-PLATE coordinates, not CAD-assembly coordinates.
     Bambu Studio centres each object on its own bbox and then places it on the
     plate, so the CAD origin -- the thing that would have told us how the parts
     bolt together -- is gone. (`connectf.stl` is the exception: it still carries
     the assembly frame, see below.)

So the assembly had to be recovered from the geometry, and this script bakes the
recovered pose into the mesh itself. That keeps SkyGrip_full.xml readable: every
new geom is placed with a plain offset and no quaternion, because the STL is
already written in the frame the XML expects.

`left clamp.3mf` and `right clamp.stl` are the SAME part (identical vertex sets,
not mirrored) -- one printed paddle used on both sides, flipped. That is why both
output claws come from a single input mesh with two different rotations.

The frame the XML expects (unchanged from the old gripper, so nothing downstream
has to move)
------------------------------------------------------------------------------
gripper_assembly body frame:
    +x  jaw closing axis (clamp_1 sits on +x, clamp_2 on -x)
    -y  approach direction -- "down" out of the gripper, toward the object
    +z  jaw width (across the paddles)
Origin: centre of the base's mounting face, i.e. the plane that sits on connectf.

How the parts map into it
-------------------------
gripper-base.3mf needs NO rotation: its own object frame already is the assembly
frame.
    base +x (48.3 mm, rack travel direction) -> +x   (closing)
    base +y (32.3 mm, toward the arm)        -> +y   (up)
    base +z (36.6 mm, across the gripper)    -> +z

Three things line up on this and it is worth recording why, because two earlier
orientations looked plausible and were not:

  * The 836 mm^2 flat face at base y = +16 is by a wide margin the largest plane
    on the part, and it comes out as the TOP face -- both the plate the racks
    ride on and the face that meets connectf. The closing axis lies in it, which
    it must: a plate whose normal is the travel direction is not something a
    rack can slide along.
  * base x is the only dimension long enough to be the travel axis -- 48.3 mm of
    body for 32 mm of paddle travel. At full aperture the outermost paddle face
    reaches x = 22.9 mm against the body's 24.2 mm half-width, i.e. just inside.
    (Putting the closing axis on base y instead leaves the body 16.2 mm to a
    side and the paddles hang out in free space at full open.)
  * The underside is an open shell, so the paddles have somewhere to retract
    into, and the base drops onto connectf with ZERO interpenetration -- even
    connectf's screw post clears. No other orientation manages that; they all
    leave the post buried in solid material.

claw local axes -> assembly axes:
    claw +y (33.3 mm, finger length)        -> -y   (points down)
    claw -x (the 265 mm^2 flat face)        -> toward the other claw
    claw +z (11.5 mm, tab-offset direction) -> +-z
The two paddles use the same part rotated differently (180 deg about x for the
+x paddle, 180 deg about z for the -x one). That is what puts their mounting tabs
at OPPOSITE z -- claw A's tab at z = -11.5, claw B's at +11.5 -- so the two racks
pass each other instead of colliding. The asymmetric tabs on an otherwise
z-symmetric part are the giveaway that this is the intended arrangement.

Units: inputs are millimetres (connectf.stl is 0.1 mm -- see SCALE_CONNECTF).
Outputs are written in millimetres, so the XML uses scale="0.001 0.001 0.001".

Run:  python prepare_gripper_meshes.py
"""

import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))

# connectf.stl is the one new part that kept its CAD frame: its Joint_2 bearing
# bolt circle (five r=1.0 holes on a r=8 circle, both side walls) is centred
# exactly on the mesh origin, identical to the old connect.stl. It is drawn in
# 0.1 mm units -- at 0.0001 it comes out 42.0 mm wide, matching old connect.stl's
# 42.0 mm to the micron. So connectf drops straight into the `connect` body with
# no offset and no rotation; only the asset scale changes.
SCALE_CONNECTF = 0.0001

# Paddle travel, per paddle, in mm. The Pololu Micro Gripper Kit is a rack and
# pinion with opposing racks and a 32 mm range of motion between the paddles;
# the paddles stay parallel through the whole range. 32 mm of GAP is 16 mm of
# travel each, which is what the slide joints get.
PADDLE_TRAVEL_MM = 16.0


def load_part(filename):
    """Load a slicer file in its own object frame, bbox-centred.

    For 3MF, trimesh returns a Scene whose graph holds the build-plate placement;
    taking the geometry directly gives the object frame and drops the plate
    transform. For the STL the plate offset is still baked into the vertices, so
    it is removed by re-centring -- which is exactly what the 3MF object frame
    already is, so both paths end up in the same frame.
    """
    obj = trimesh.load(os.path.join(HERE, filename))
    mesh = list(obj.geometry.values())[0] if isinstance(obj, trimesh.Scene) else obj
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
    return mesh


def rotate(mesh, image_of_x, image_of_y, image_of_z):
    """Rigidly rotate by naming where the local axes end up. Refuses reflections."""
    R = np.array([image_of_x, image_of_y, image_of_z], float).T
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0):
        raise ValueError(f"not a proper rotation (det={det:+.3f}) -- that is a mirror")
    out = mesh.copy()
    out.vertices = out.vertices @ R.T
    return out


X = np.array([1.0, 0, 0])
Y = np.array([0, 1.0, 0])
Z = np.array([0, 0, 1.0])


def main():
    base_raw = load_part("gripper-base.3mf")
    claw_raw = load_part("left clamp.3mf")

    # The two supplied claw files are the same geometry; assert it rather than
    # trust it, because the whole two-rotations-of-one-part scheme rests on it.
    other = load_part("right clamp.stl")
    same = np.allclose(np.sort(claw_raw.vertices, axis=0),
                       np.sort(other.vertices, axis=0), atol=1e-3)
    print(f"left clamp.3mf and right clamp.stl are the same part: {same}")

    # ---- base -------------------------------------------------------------
    # 180 deg about the closing axis: the part's object frame is the assembly
    # frame turned upside down.
    base = rotate(base_raw, image_of_x=X, image_of_y=-Y, image_of_z=-Z)
    # Origin at the centre of the mounting face: x/z stay bbox-centred, y is
    # shifted so the top of the base (the face that meets connectf) is y = 0 and
    # the whole gripper hangs in -y from there.
    base.vertices[:, 1] -= base.bounds[1][1]
    base.export(os.path.join(HERE, "gripper_base.stl"))

    # ---- claws ------------------------------------------------------------
    # The gripping face is the 265 mm^2 flat plane at claw x = -2.5. Both claws
    # are shifted so that face lands on x = 0, so at ctrl = 0 the two faces are
    # coincident (gap 0) and the joint value IS the half-gap in metres.
    face_x = -2.5

    # clamp_1, the +x paddle: 180 deg about x. Its flat face ends up pointing -x,
    # i.e. inward at the other paddle.
    clamp_1 = rotate(claw_raw, image_of_x=X, image_of_y=-Y, image_of_z=-Z)
    clamp_1.vertices[:, 0] -= face_x            # face to x = 0, body on +x
    clamp_1.vertices[:, 1] -= clamp_1.bounds[1][1]   # top of the claw to y = 0

    # clamp_2, the -x paddle: 180 deg about z. Same part, flipped, so its tabs
    # land on the opposite rack channel.
    clamp_2 = rotate(claw_raw, image_of_x=-X, image_of_y=-Y, image_of_z=Z)
    clamp_2.vertices[:, 0] += face_x            # face to x = 0, body on -x
    clamp_2.vertices[:, 1] -= clamp_2.bounds[1][1]

    clamp_1.export(os.path.join(HERE, "right_clamp.stl"))
    clamp_2.export(os.path.join(HERE, "left_clamp.stl"))

    for name, mesh in [("gripper_base", base), ("right_clamp (+x)", clamp_1),
                       ("left_clamp  (-x)", clamp_2)]:
        lo, hi = np.round(mesh.bounds, 3)
        print(f"{name:18s} x=[{lo[0]:7.2f},{hi[0]:7.2f}] "
              f"y=[{lo[1]:7.2f},{hi[1]:7.2f}] z=[{lo[2]:7.2f},{hi[2]:7.2f}]  mm")

    # ---- where the claws sit under the base -------------------------------
    # The base is an open shell underneath, so the paddles retract UP into it
    # rather than butting against a floor: they are hung with ENGAGE_MM of their
    # length inside the body, which is where the racks run, leaving the rest
    # protruding as the working jaw. The exposed length is what sets grasp_site
    # and the pad boxes, so it is stated here rather than left implicit.
    ENGAGE_MM = 13.3
    mount_y = base.bounds[0][1] + ENGAGE_MM
    exposed = base.bounds[0][1] - (mount_y + clamp_1.bounds[0][1])
    print(f"\nbase spans y                       : {base.bounds[0][1]:.3f} .. 0.000 mm")
    print(f"clamp body y (paddle top)          : {mount_y:.3f} mm")
    print(f"claw tip reaches                   : {mount_y + clamp_1.bounds[0][1]:.3f} mm")
    print(f"paddle protruding below the body   : {exposed:.3f} mm")
    print(f"grasp_site y (mid of exposed jaw)  : {base.bounds[0][1] - exposed / 2:.4f} mm")
    print(f"pad centre in CLAMP-local y        : {base.bounds[0][1] - exposed / 2 - mount_y:.4f} mm")

    # ---- how high the base sits on connectf -------------------------------
    # Solved rather than guessed: slide the base along the connect +z axis until
    # it stops intersecting connectf, and take the first clear height. connectf
    # is watertight, so "inside" is well defined; the base is not, so it is
    # tested as a dense point sample of its surface.
    connectf = trimesh.load(os.path.join(HERE, "connectf.stl"), force="mesh")
    connectf.vertices = connectf.vertices * (SCALE_CONNECTF * 1000.0)  # -> mm
    pts = base.sample(20000)
    # assembly -> connect frame: A x -> c x, A y -> -c z, A z -> c y
    probe = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])
    print(f"\nconnectf top of stack              : {connectf.bounds[1][2]:.2f} mm")
    for pz in np.arange(26.0, 46.01, 0.5):
        inside = connectf.contains(probe + np.array([0.0, 0.0, pz]))
        n = int(inside.sum())
        if n == 0:
            print(f"lowest clash-free mount height pz  : {pz:.2f} mm (connect frame)")
            break
        if abs(pz - round(pz)) < 1e-9 and pz % 2 == 0:
            print(f"   pz={pz:5.1f} mm -> {n:5d}/20000 sample points inside connectf")
    else:
        print("no clash-free height found in the search range")


if __name__ == "__main__":
    main()
