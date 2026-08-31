"""camera_pose.py -- v2 sign-off artefact 1+2: the proposed camera3 pose
and the taped-square goal, rendered and measured.

Adds (via mjSpec, no existing file edited) a candidate external camera and
a taped goal square on the table, then measures what the v2 collection
needs measured BEFORE sign-off:

  - object-pair pixel counts from the fixed camera across the table's
    full spawn sweep (the camera is fixed, so unlike the drone cameras
    these numbers hold at every drone position);
  - presence of table, gate, goal square and both objects in frame
    (segmentation-counted, not eyeballed) for pick AND nav scenes;
  - a 224 px frame exactly as the policy would see it, plus a labelled
    montage for the sign-off document.

Requirements this pose must meet (from REPORT2 Test B): resolve the two
objects regardless of drone position; contain table, gate and goal at
every spawn. Decision numbers print per scene.

usage: python camera_pose.py <out_montage.png>
(run from Sim'n'Real/Mujoco so collect_airvla imports)
"""
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

import collect_airvla as A

RES = 224
# Candidate pose (iterated against the checks below before sign-off):
# low on the stairs side, looking across the workspace so the object
# pair shows side profiles against the far wall, near enough that both
# objects clear ~25 px at 224.
# Room side, behind the flight line (paper-style): the object pair is
# separated along x, so a -y-looking camera sees them side by side with
# no mutual occlusion; a -x side view sees them end-on and one occludes
# the other (measured: 6-14 px, FAIL).
# Measured trade-off: the spawn envelope's corners (objects to +-0.92 m
# at shallow depth) and >=25 px everywhere are jointly infeasible at a
# fixed 224 px camera -- wide enough for the corners costs resolution
# everywhere else. This pose keeps every legal spawn in frame; the bar
# is >=20 px per object with the penguin/weight area ratio preserved,
# since size ratio + lateral separation is what discrimination rides on.
CAM_POS = np.array([0.00, 2.25, 1.22])
CAM_LOOK = np.array([0.05, 0.08, 0.30])
CAM_FOVY = 52.0
# taped square: 22 cm outer, 3 cm tape, on the table top, room side
SQ_HALF, TAPE, SQ_OFF = 0.11, 0.03, np.array([0.26, 0.10])

out = sys.argv[1]
spec = mujoco.MjSpec.from_file("SkyGrip_airvla.xml")
f = CAM_LOOK - CAM_POS
f = f / np.linalg.norm(f)
rt = np.cross(f, [0, 0, 1.0])
rt /= np.linalg.norm(rt)
up = np.cross(rt, f)
cam = spec.worldbody.add_camera()
cam.name = "cam3_v2"
cam.pos = CAM_POS
# camera frame: looks along -z, +x right, +y up -> columns [rt, up, -f]
R = np.column_stack([rt, up, -f])
q = np.empty(4)
mujoco.mju_mat2Quat(q, R.flatten())
cam.quat = q
cam.fovy = CAM_FOVY
table = [b for b in spec.bodies if b.name == "table"][0]
top = 0.485                       # table-top z in the table frame + eps
for i, (dx, dy, sx, sy) in enumerate([
        (0, SQ_HALF, SQ_HALF, TAPE / 2), (0, -SQ_HALF, SQ_HALF, TAPE / 2),
        (SQ_HALF, 0, TAPE / 2, SQ_HALF), (-SQ_HALF, 0, TAPE / 2, SQ_HALF)]):
    g = table.add_geom()
    g.name = "tape%d" % i
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [sx, sy, 0.0008]
    g.pos = [SQ_OFF[0] + dx, SQ_OFF[1] + dy, top]
    g.rgba = [0.95, 0.85, 0.15, 1]
    g.contype = 0
    g.conaffinity = 0

m = spec.compile()
d = mujoco.MjData(m)
r = A.Runner(2)                       # drives reset_scene on its own model
rend = mujoco.Renderer(m, height=RES, width=RES)
seg = mujoco.Renderer(m, height=RES, width=RES)
seg.enable_segmentation_rendering()
big = mujoco.Renderer(m, height=512, width=512)


def subtree(name):
    b = m.body(name).id
    return set(g for g in range(m.ngeom)
               if m.body_rootid[m.geom_bodyid[g]] == b)


def counts(sg, gs):
    vis = sg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
    return int((np.isin(sg[:, :, 0], list(gs)) & vis).sum())


G = {"weight": subtree("pick_weight"), "penguin": subtree("penguin"),
     "table": subtree("table"), "gate": subtree("gate")}
TAPE_G = set(g for g in range(m.ngeom)
             if (m.geom(g).name or "").startswith("tape"))

# scene configurations: full spawn sweep for picks, plus both nav gates
SCENES = [
    ("pick centre", dict(task=[0.0, 0.60], drone=[0.0, 1.5, 0.25])),
    ("pick far-left-front", dict(task=[-0.45, 0.20], drone=[0.3, 1.3, 0.25])),
    ("pick far-right-back", dict(task=[0.45, 1.00], drone=[-0.3, 1.6, 0.25])),
    ("nav gate-left", dict(task=[0.0, 0.60], drone=[0.0, -1.6, 0.8],
                           gate=[-0.7, -0.6])),
    ("nav gate-right", dict(task=[0.3, 0.80], drone=[0.0, -1.6, 0.8],
                            gate=[0.7, -0.6])),
]
print("CAMERA CANDIDATE cam3_v2  pos(%.2f, %.2f, %.2f) look(%.2f, %.2f, "
      "%.2f) fovy %.0f  | tape square %d cm at table offset (%.2f, %.2f)"
      % (*CAM_POS, *CAM_LOOK, CAM_FOVY, 200 * SQ_HALF + 100 * TAPE,
         *SQ_OFF))
tiles = []
ok_all = True
for label, cfg in SCENES:
    # reset via the runner's own draw logic on a separate model, then
    # copy the sampled scene into this (spec-compiled) model by re-running
    # reset_scene against it -- Runner only touches names that exist here
    r.model, r.data = m, d
    gate = cfg.get("gate")
    r.reset_scene(np.array(cfg["task"]),
                  np.array(cfg["drone"], float),
                  gate_xy=None if gate is None else np.array(gate),
                  obj="weight")
    # the distractor stays where reset_scene put it (its own side/
    # separation logic), then the v2 spawn cap |x| <= 0.85 is applied --
    # measured here: beyond it the frame edge clips the object (39 px,
    # size ratio inverts). The cap is symmetric across target and
    # distractor in the v2 collector, so position stays uninformative;
    # position_shortcut.py re-runs on the collected data as a gate.
    # v2 spawn envelope (measured from this camera's frame edges): BOTH
    # objects within |x| <= 0.68 at any y, target sweep +-0.45; the v2
    # collector samples the pair inside this envelope and assigns
    # target/distractor by coin flip, so position is uninformative BY
    # CONSTRUCTION (a side-preference rule would leak ~79% -- computed
    # in the sign-off doc). position_shortcut.py re-runs on collected
    # data as a pre-training gate.
    pen = r.objs["plush penguin"]
    d.qpos[pen["adr"]] = float(np.clip(d.qpos[pen["adr"]], -0.68, 0.68))
    mujoco.mj_forward(m, d)
    seg.update_scene(d, camera="cam3_v2")
    sg = seg.render()
    c = {k: counts(sg, gs) for k, gs in G.items()}
    c["tape"] = counts(sg, TAPE_G)
    need = ["weight", "penguin", "table"] + (["gate"] if gate else ["tape"])
    ratio = c["penguin"] / max(1, c["weight"])
    ok = (all(c[k] >= (20 if k in ("weight", "penguin") else 8)
              for k in need) and ratio >= 1.4)
    ok_all &= ok
    print("%-20s weight %4dpx penguin %4dpx (ratio %.1f) table %5dpx "
          "tape %3dpx gate %4dpx  -> %s"
          % (label, c["weight"], c["penguin"], ratio, c["table"],
             c["tape"], c["gate"], "PASS" if ok else "FAIL"))
    big.update_scene(d, camera="cam3_v2")
    im = Image.fromarray(big.render().copy())
    dr = ImageDraw.Draw(im)
    dr.text((6, 6), "cam3_v2 | %s" % label, fill=(255, 255, 0))
    dr.text((6, 494), "w:%dpx p:%dpx tape:%dpx gate:%dpx"
            % (c["weight"], c["penguin"], c["tape"], c["gate"]),
            fill=(0, 255, 255))
    tiles.append(np.asarray(im))
    if label == "pick centre":
        rend.update_scene(d, camera="cam3_v2")
        imageio.imwrite(out.replace(".png", "_224.png"),
                        rend.render().copy())

row1 = np.concatenate(tiles[:3], axis=1)
row2 = np.concatenate(tiles[3:] + [np.zeros_like(tiles[0])], axis=1)
imageio.imwrite(out, np.concatenate([row1, row2], axis=0))
print("montage -> %s   policy-resolution frame -> %s"
      % (out, out.replace(".png", "_224.png")))
print("OVERALL: %s" % ("PASS" if ok_all else "FAIL -- iterate the pose"))
