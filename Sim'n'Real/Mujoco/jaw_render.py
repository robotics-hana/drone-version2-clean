"""Render the jaws at the grasp pose so the geometry can be SEEN, not inferred.

Every attempt this session to reason about where the fingers are from body/geom
origins has been wrong, because the clamp meshes are asymmetric and their frame
origins sit nowhere near the surfaces that touch the object. Pictures settle it.
"""
import numpy as np
import mujoco

M = "SkyGrip_full.xml"


def render(open_cmd, site_z, path, cam_pos, cam_xy):
    model = mujoco.MjModel.from_xml_path(M)
    data = mujoco.MjData(model)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    qa = model.jnt_qposadr[model.body_jntadr[ob]]

    mujoco.mj_resetData(model, data)
    obj = data.qpos[qa:qa + 3].copy()
    data.ctrl[aid("act_joint1")] = 0.0
    data.ctrl[aid("act_joint2")] = 0.0
    data.ctrl[aid("act_gripper")] = open_cmd
    data.qpos[9] = open_cmd
    data.qpos[10] = -open_cmd
    data.qpos[0:3] = [obj[0], obj[1], 1.0]
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    data.qpos[0:3] += (np.array([obj[0], obj[1], site_z]) - data.site_xpos[sid])
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [obj[0], obj[1], site_z]
    cam.distance = 0.30
    cam.azimuth = cam_pos
    cam.elevation = cam_xy

    with mujoco.Renderer(model, 480, 640) as r:
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        # Site spheres are drawn at a fixed screen-independent scale and swamp a
        # 0.30 m close-up -- the grasp_site marker alone hides the whole gripper.
        for g in range(len(opt.sitegroup)):
            opt.sitegroup[g] = 0
        r.update_scene(data, camera=cam, scene_option=opt)
        img = r.render()
    try:
        import imageio.v3 as iio
        iio.imwrite(path, img)
    except Exception:
        from PIL import Image
        Image.fromarray(img).save(path)
    print(f"  wrote {path}  site={data.site_xpos[sid]}  obj={data.xpos[ob]}")


if __name__ == "__main__":
    for label, z in (("centre", 0.540), ("low", 0.500)):
        for ang, az, el in (("front", 90, 0), ("side", 0, 0)):
            render(0.016, z, f"jaw_{label}_{ang}_open.png", az, el)
            render(0.000, z, f"jaw_{label}_{ang}_shut.png", az, el)
