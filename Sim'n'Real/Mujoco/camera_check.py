"""
Render every camera in SkyGrip_full.xml across several arm poses.

Produces:
  camera_grid.png      - contact sheet, one row per camera, one column per arm pose
  cam_<name>.png       - full-res single shot of each camera at the mid ("reach") pose

The model has no target_object body, so the `current_target` mocap sphere is
parked just ahead of the jaws each pose to stand in for a graspable object --
if you can't see the sphere in a given camera, that camera can't see the grasp.

    python camera_check.py
"""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

MODEL = "SkyGrip_full.xml"
H, W = 360, 480

# (label, Joint_1, Joint_2) -- Joint_1 spans -1.6..1.6. POSITIVE angles swing the arm
# up-and-forward (+y), which is the half of the range scene_cam looks into; negative
# angles swing it to -y, away from scene_cam. Sample the forward half.
POSES = [
    ("retracted", 0.0, 0.0),
    ("fwd-mid", 1.0, 0.6),
    ("fwd-high", 1.5, 1.2),
]


def grasp_point(model, data):
    """World position of the true grasp point: midpoint of the visible clamp geoms.

    NOT the gripper_assembly body origin -- the clamp meshes carry ~7 cm offsets,
    so the body origin sits well behind where the jaws actually close.
    """
    ids = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] in (model.body("clamp_1").id, model.body("clamp_2").id)
    ]
    return np.mean([data.geom_xpos[g] for g in ids], axis=0)


def set_pose(model, data, j1, j2):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.5
    data.qpos[model.joint("Joint_1").qposadr[0]] = j1
    data.qpos[model.joint("Joint_2").qposadr[0]] = j2
    mujoco.mj_forward(model, data)

    # Probe A ("grasp"): stand-in object just beyond the jaws, along the -y approach axis.
    #   Tests the manipulation close-up -- can this camera see what's being grasped?
    gid = model.body("gripper_assembly").id
    R = data.xmat[gid].reshape(3, 3)
    mid = model.body("current_target").mocapid[0]
    if mid >= 0:
        data.mocap_pos[mid] = grasp_point(model, data) + R @ np.array([0.0, -0.03, 0.0])

    # Probe B ("approach"): target sitting on the ground ahead of the drone.
    #   Tests the navigation view -- can this camera see the target during approach?
    #   Judging a forward-looking camera solely on Probe A is unfair: it is doing a
    #   different job, and failing Probe A is expected rather than a defect.
    m_id = model.body("waypoint_0").mocapid[0]
    if m_id >= 0:
        data.mocap_pos[m_id] = np.array([0.0, -0.45, 0.05])

    m_id = model.body("waypoint_1").mocapid[0]
    if m_id >= 0:
        data.mocap_pos[m_id] = np.array([0.0, 0.0, -5.0])   # park unused marker out of frame
    mujoco.mj_forward(model, data)


def visibility(model, data, cam_id, target, exclude="current_target"):
    """(in_fov, unoccluded) for `target` from camera `cam_id`.

    `target` must be the stand-in object position (free space just ahead of the
    jaws), NOT the grasp midpoint. The midpoint sits behind the gripper's own
    housing from most angles, so ray-casting to it reports a false occlusion for
    every camera -- including ones that plainly show the jaws in the render.
    The current_target body is excluded so the marker doesn't occlude itself.
    """
    pos = data.cam_xpos[cam_id]
    fwd = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]   # MuJoCo cameras look down -Z
    vec = target - pos
    dist = np.linalg.norm(vec)
    if dist < 1e-9:
        return False, False
    v = vec / dist
    ang = np.degrees(np.arccos(np.clip(fwd @ v, -1.0, 1.0)))

    hit = np.zeros(1, dtype=np.int32)
    frac = mujoco.mj_ray(model, data, pos, v, None, 1, model.body(exclude).id, hit)

    # "Unoccluded" means nothing OTHER than the gripper itself stands in the way.
    # Requiring the ray to reach the target point outright gives false negatives:
    # the probe sits in free space just beyond the jaws, so the gripper's own
    # housing legitimately intercepts the ray in views that plainly show it.
    reached = frac < 0 or frac > dist - 0.006
    if not reached and hit[0] >= 0:
        blocker = model.body(model.geom_bodyid[int(hit[0])]).name
        reached = blocker in ("clamp_1", "clamp_2", "gripper_assembly")
    return ang < model.cam_fovy[cam_id] / 2.0, reached


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)

    cams = [model.camera(i).name for i in range(model.ncam)]
    print(f"{model.ncam} cameras: {', '.join(cams)}\n")

    sheet = Image.new("RGB", (W * len(POSES), H * len(cams)))
    draw = ImageDraw.Draw(sheet)

    for row, cam in enumerate(cams):
        cam_id = model.camera(cam).id
        for col, (label, j1, j2) in enumerate(POSES):
            set_pose(model, data, j1, j2)
            grasp_obj = data.mocap_pos[model.body("current_target").mocapid[0]]
            appr_obj = data.mocap_pos[model.body("waypoint_0").mocapid[0]]
            g_fov, g_los = visibility(model, data, cam_id, grasp_obj)
            a_fov, a_los = visibility(model, data, cam_id, appr_obj,
                                      exclude="waypoint_0")

            renderer.update_scene(data, camera=cam)
            frame = renderer.render().copy()
            sheet.paste(Image.fromarray(frame), (col * W, row * H))
            draw.text((col * W + 6, row * H + 6),
                      f"{cam} | {label} | grasp={'Y' if (g_fov and g_los) else 'n'}"
                      f" approach={'Y' if (a_fov and a_los) else 'n'}",
                      fill=(255, 255, 0))

            if col == 1:  # save a full-res shot at the middle pose
                Image.fromarray(frame).save(f"cam_{cam}.png")

            print(f"  {cam:<12} {label:<9} "
                  f"grasp={'yes' if (g_fov and g_los) else 'NO ':<3} "
                  f"approach={'yes' if (a_fov and a_los) else 'NO'}")

    sheet.save("camera_grid.png")
    print(f"\nwrote camera_grid.png ({len(cams)} cameras x {len(POSES)} poses)")
    print("wrote cam_<name>.png for each camera at the 'reach' pose")


if __name__ == "__main__":
    main()
