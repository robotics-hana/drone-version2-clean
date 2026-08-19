"""Which arm pose should the grasp use, and which objects can stand up?

Two questions that only make sense together:

  1. Standing an object up raises its grasp point off the mat, which is worth a
     lot -- picking a flat object off the floor needs the body at ~0.23 m with
     only ~0.08 m of leg clearance, and that is the configuration that used to
     flip the drone. But an object only counts as "upright" if it would actually
     STAY upright, so that is screened rather than assumed.

  2. The arm pose sets how far below the body the jaws hang. A deeper drop means
     the body flies HIGHER for the same object, which buys leg clearance -- but
     the deep poses are not free: they swing the arm's mass sideways (roll
     disturbance on the axis the flight controller is weakest about) and tilt the
     jaw mouth off vertical, so the jaws no longer descend squarely onto an
     upright object.

So this sweeps the whole (Joint_1, Joint_2) envelope, scores each pose, and then
for every stable object works out the body altitude and leg clearance it implies.

Run:  python arm_reach_study.py
"""

import os

import numpy as np
import mujoco

import collect_demos as C

HERE = os.path.dirname(os.path.abspath(__file__))
LEG_DROP = 0.15          # legs hang this far below the body (SkyGrip_full.xml)
MIN_LEG_CLEAR = 0.08     # below this a small attitude error puts a leg on the mat

# From prepare_objects.py -- the 7 YCB objects a 32 mm jaw can close on, WITH the
# pose that made them graspable. Stability must be judged in that same pose:
# judging it as-scanned while grasping upright mixes two different objects (a
# spatula lying down is 33 mm tall and rock solid; standing on end it is 299 mm
# and falls over if you look at it).
import trimesh as _tm
_R = {"as-scanned": np.eye(3),
      "upright-x": _tm.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3],
      "upright-y": _tm.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]}
GRASPABLE = {
    "031_spoon": "upright-y", "033_spatula": "upright-y",
    "037_scissors": "upright-y", "026_sponge": "upright-x",
    "073-a_lego_duplo": "upright-y", "040_large_marker": "upright-x",
    "006_mustard_bottle": "as-scanned",
}


def upright_stability(mesh):
    """Tip angle of the object standing on its base, and its grasp height.

    Tip angle = atan(base half-width / CoM height): how far it can lean before
    its centre of mass leaves the footprint. A pencil standing on end has a tiny
    one; a bottle has a large one. Cheap, and it is the quantity that decides
    whether the thing survives being set down on a mat.
    """
    m = mesh.copy()
    m.vertices -= m.bounds[0]
    top = float(m.bounds[1][2])
    base = m.section(plane_origin=[0, 0, 0.02 * top], plane_normal=[0, 0, 1])
    if base is None:
        return None
    p = base.vertices
    half = 0.5 * min(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    com_z = float(m.centroid[2] - m.bounds[0][2])
    if com_z <= 0:
        return None
    return dict(top_m=top, half_base_m=float(half), com_z_m=com_z,
                tip_deg=float(np.degrees(np.arctan2(half, com_z))))


def arm_envelope(model):
    """Sweep (j1, j2) and record what each pose gives the grasp."""
    ik = C.ArmIK(model)
    rows = []
    for (j1, j2), off, dy, mouth in zip(ik._grid, ik._offsets, ik._com_dy, ik._mouthz):
        rows.append(dict(j1=float(j1), j2=float(j2),
                         drop=float(-off[2]), lateral=float(off[1]),
                         mouth=float(mouth), com_dy=float(abs(dy))))
    return rows


def main():
    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "SkyGrip_full.xml"))
    rows = arm_envelope(model)
    drops = np.array([r["drop"] for r in rows])
    mouths = np.array([r["mouth"] for r in rows])
    comdy = np.array([r["com_dy"] for r in rows])

    print("ARM ENVELOPE  (Joint_1, Joint_2 swept over their full range)")
    print(f"  vertical drop, body -> jaws : {drops.min():.3f} .. {drops.max():.3f} m")
    print(f"  poses with a near-vertical mouth (|cos| > 0.98): "
          f"{(mouths > 0.98).sum()} of {len(rows)}")
    print()
    print("  candidate grasp poses:")
    print(f"    {'name':22s} {'j1':>6s} {'j2':>6s} {'drop':>7s} {'mouth':>6s} {'armCoM dy':>10s}")
    picks = {}
    # deepest pose that still points the jaws straight down
    vert = [r for r in rows if r["mouth"] > 0.98]
    if vert:
        picks["straight down, deepest"] = max(vert, key=lambda r: r["drop"])
        picks["straight down, gentlest"] = min(vert, key=lambda r: r["com_dy"])
    # what the model uses today
    q, off = C.ArmIK(model).solve_drop(0.19)
    picks["current (solve_drop 0.19)"] = dict(j1=q[0], j2=q[1], drop=-off[2],
                                              lateral=off[1], mouth=np.nan,
                                              com_dy=abs(off[0]))
    # a folded, shallow pose -- jaws tucked close under the body
    shallow = [r for r in rows if r["mouth"] > 0.95 and r["drop"] < 0.14]
    if shallow:
        picks["folded / shallow"] = min(shallow, key=lambda r: r["com_dy"])
    for name, r in picks.items():
        print(f"    {name:22s} {r['j1']:6.2f} {r['j2']:6.2f} {r['drop']:6.3f}m "
              f"{r['mouth']:6.2f} {r['com_dy']*1000:8.1f}mm")

    # --- objects -------------------------------------------------------
    import trimesh
    print("\nUPRIGHT STABILITY of the graspable set "
          f"(tip angle; >20 deg is comfortably stable)")
    print(f"  {'object':24s} {'height':>8s} {'tip':>7s} {'verdict':22s}")
    stable = []
    for name, pose in GRASPABLE.items():
        path = os.path.join(HERE, "ycb_assets", name, "google_16k", "textured.obj")
        if not os.path.exists(path):
            print(f"  {name:24s}   not downloaded -- run prepare_objects.py")
            continue
        mm = trimesh.load(path, force="mesh")
        mm.vertices = mm.vertices @ _R[pose].T          # judge in the GRASPABLE pose
        s = upright_stability(mm)
        if s is None:
            continue
        v = ("STABLE" if s["tip_deg"] > 20 else
             "marginal" if s["tip_deg"] > 12 else "TOPPLES -- lay it down")
        if s["tip_deg"] > 20:
            stable.append((name, s))
        print(f"  {name:24s} {s['top_m']*1000:7.0f}mm {s['tip_deg']:6.1f}d  {v}   [{pose}]")

    # --- combine -------------------------------------------------------
    if stable and picks:
        print("\nBODY ALTITUDE AND LEG CLEARANCE, per stable object x arm pose")
        print(f"  (grasp 13 mm below the object top; legs hang {LEG_DROP*1000:.0f} mm; "
              f"want > {MIN_LEG_CLEAR*1000:.0f} mm clear)")
        print(f"  {'object':22s} {'pose':24s} {'body z':>8s} {'legs':>8s}")
        for name, s in stable:
            gz = s["top_m"] - 0.013
            for pname, r in picks.items():
                body = gz + r["drop"]
                clear = body - LEG_DROP
                flag = "" if clear > MIN_LEG_CLEAR else "   <-- too low"
                print(f"  {name:22s} {pname:24s} {body*1000:7.0f}mm "
                      f"{clear*1000:7.0f}mm{flag}")


if __name__ == "__main__":
    main()
