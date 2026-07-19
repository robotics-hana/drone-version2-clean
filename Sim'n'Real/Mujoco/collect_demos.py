"""
SkyGrip demo collection -> LeRobotDataset (v3.0 API)

Scripts a pick-and-place trajectory per episode, randomises object pose,
drone start pose and lighting, and records wrist + scene frames, state and
action to a LeRobotDataset for SmolVLA finetuning.

Run:
    python collect_demos.py --episodes 50 --out_repo_id yourname/skygrip_pick_place


ACTION SPACE (6-D)
------------------
    [drone_x, drone_y, drone_z, joint1, joint2, gripper]

The VLA emits SETPOINTS at chunk rate; MPPI realises them at control rate.
It does NOT emit thrust/torque. Three reasons:

  1. Thrust and body torques are only meaningful relative to instantaneous
     attitude and mass. A chunk of them replayed even slightly out of phase
     is not just degraded, it is unstable -- exactly the discontinuity
     problem RTC exists to paper over.
  2. Setpoints are what the platform's own controller consumes. AirVLA does
     the same thing, emitting position setpoints to PX4; here MPPI plays the
     role PX4 plays there, with the advantage that it optimises over the
     coupled drone+arm dynamics rather than needing a payload-compensation
     patch bolted on at inference time.
  3. Joint-space for the manipulator is the dissertation's stated choice.
     The drone body has no joints, so its "joint space" IS its pose -- a
     Cartesian body setpoint does not contradict that decision, whereas an
     end-effector-space target for the ARM would.

The gripper channel is continuous here (metres of jaw travel) because that
is what the actuator takes, but it is driven as an effectively binary
open/closed phase decision. If the abstain head later needs a discrete
grasp token, that is a separate output, not a reinterpretation of this one.


OBSERVATION STATE (16-D)
------------------------
    [joint1, joint2, gripper, pos(3), quat(4), linvel(3), angvel(3)]

Drone pose and twist ARE included. This was flagged as an open design
question, and the answer follows from the geometry: the gripper's world
position is a function of drone pose AND arm angles, so arm angles alone are
not a Markov state for a manipulation task on a moving base. A fixed-base
SmolVLA setup gets away with joints-only precisely because its base pose is
constant. Ours is not.

Cost: smolvla_base was pretrained with state dim 6. Anything up to 32 is
accepted and zero-padded (pad_vector pads but never truncates -- 33 would
raise), but the further from 6, the less the pretrained input projection
transfers. 16 is a deliberate trade, not an oversight.


CAMERAS
-------
Keys MUST be observation.images.camera1/camera2/... . smolvla_base ships
non-empty input_features naming camera1/camera2/camera3, and factory.py only
infers feature names from the dataset `if not cfg.input_features:` -- which
is false for this checkpoint. Dataset visuals must therefore be a subset of
{camera1, camera2, camera3}. Naming them "wrist"/"scene" fails
validate_visual_features_consistency outright. The mapping to physical
cameras is recorded in CAMERAS below.


STILL BLOCKED
-------------
ScriptedController.step() is a stub. The arm and gripper are position
actuators so the FSM can drive them directly, but flying the body to each
waypoint needs the MPPI controller wired in, and the phase transitions need
a 2-link IK that has not been written. Until then this script runs and
produces a correctly-shaped dataset, but the drone does not fly -- do not
mistake a successful run for usable demonstrations.
"""

import argparse
import numpy as np
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDataset

MODEL_PATH = "SkyGrip_full.xml"
FPS = 30

# SmolVLA resizes to 512x512 internally (resize_imgs_with_padding). Logging
# smaller just means upscaling later with no detail to recover; 480x640 keeps
# real resolution without bloating the dataset.
IMG_H, IMG_W = 480, 640

# LeRobot feature key -> MuJoCo camera name. The keys are fixed by the
# checkpoint (see CAMERAS note above); the values are ours to choose.
CAMERAS = {
    "camera1": "wrist_cam",    # manipulation close-up, top-down over the jaws
    "camera2": "scene_cam",    # forward/navigation view from the nose
}

PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE"]

STATE_NAMES = (
    ["joint1", "joint2", "gripper"]
    + [f"drone_pos_{a}" for a in "xyz"]
    + [f"drone_quat_{a}" for a in "wxyz"]
    + [f"drone_linvel_{a}" for a in "xyz"]
    + [f"drone_angvel_{a}" for a in "xyz"]
)
ACTION_NAMES = ["drone_x", "drone_y", "drone_z", "joint1", "joint2", "gripper"]

# Jaw separation runs 10.7 mm (closed) to 84.6 mm (fully open); the target
# cube is 40 mm across, so it must be wider than ~0.015 to clear on approach.
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.012


class ScriptedController:
    """Placeholder. Replace step() with the real MPPI call.

    The arm/gripper half is real -- they are <position> actuators, so writing
    a joint target to ctrl is the actual command path, not a stand-in. Only
    the body flight is missing.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.drone_target = np.array([0.0, 0.0, 1.5])
        self.joint_target = np.zeros(2)
        self.gripper_cmd = GRIPPER_OPEN
        self.idx = {n: model.actuator(n).id
                    for n in ("act_joint1", "act_joint2", "act_gripper")}

    def set_targets(self, drone_xyz, joints, gripper):
        self.drone_target = np.asarray(drone_xyz, dtype=np.float64)
        self.joint_target = np.asarray(joints, dtype=np.float64)
        self.gripper_cmd = float(gripper)

    def step(self):
        # Arm + gripper: real command path.
        self.data.ctrl[self.idx["act_joint1"]] = self.joint_target[0]
        self.data.ctrl[self.idx["act_joint2"]] = self.joint_target[1]
        self.data.ctrl[self.idx["act_gripper"]] = self.gripper_cmd
        # TODO: body flight. Wire PureMPPIController here -- set its target_pos
        # to self.drone_target, call mppi_step(), and write the thrust/torque
        # channels. Without this the drone does not fly.

    def action(self):
        return np.concatenate([self.drone_target, self.joint_target,
                               [self.gripper_cmd]]).astype(np.float32)


def randomise_episode(model, data, rng):
    """Domain-randomise object, drone start pose and lighting.

    Volume alone overfits: LeRobot's own guidance pairs "~50 episodes" with 5
    distinct object positions x 10 episodes, and documents 25 episodes as too
    few. Randomise per episode, not per batch of episodes.
    """
    mujoco.mj_resetData(model, data)

    # --- object pose: position within reach, plus yaw so the grasp is not
    #     always axis-aligned (a fixed yaw teaches one approach angle only)
    bid = model.body("target_object").id
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    obj_xy = np.array([rng.uniform(-0.18, 0.18), rng.uniform(0.22, 0.42)])
    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    data.qpos[adr:adr + 3] = [obj_xy[0], obj_xy[1], 0.02]
    data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # --- drone start pose: vary where the episode begins, so the policy sees
    #     approach from a spread of offsets rather than one canned trajectory
    data.qpos[0:3] = [rng.uniform(-0.10, 0.10),
                      rng.uniform(-0.10, 0.10),
                      rng.uniform(1.35, 1.60)]

    # --- lighting: diffuse level and sun direction. Cheap to vary and it is
    #     what shifts most between sim and any real deployment.
    sun = model.light("sunlight").id
    fill = model.light("ambient_fill").id
    lvl = rng.uniform(0.55, 1.00)
    model.light_diffuse[sun] = [lvl, lvl, lvl]
    model.light_dir[sun] = _unit([rng.uniform(-0.4, 0.4),
                                  rng.uniform(-0.4, 0.4),
                                  -1.0])
    amb = rng.uniform(0.15, 0.40)
    model.light_diffuse[fill] = [amb, amb, amb]

    # --- object colour, so the policy keys on shape/position not one RGB
    gid = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid][0]
    model.geom_rgba[gid, :3] = rng.uniform(0.15, 0.9, size=3)

    mujoco.mj_forward(model, data)
    return np.array([obj_xy[0], obj_xy[1], 0.02])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def get_state(model, data):
    """16-D proprioceptive state. Joints looked up by name -- qpos layout has
    shifted before and will again if bodies are added."""
    j = lambda n: data.qpos[model.joint(n).qposadr[0]]
    return np.concatenate([
        [j("Joint_1"), j("Joint_2"), j("right_clamp")],
        data.qpos[0:3],      # drone position
        data.qpos[3:7],      # drone orientation (quat, wxyz)
        data.qvel[0:3],      # linear velocity
        data.qvel[3:6],      # angular velocity
    ]).astype(np.float32)


def grasp_site_pos(model, data):
    """True grasp point: the jaw midpoint, NOT the gripper_assembly body
    origin (those differ by 7.2 cm -- wider than the 40 mm target)."""
    return data.site_xpos[model.site("grasp_site").id].copy()


def run_episode(model, data, renderer, controller, rng, task_text):
    obj = randomise_episode(model, data, rng)

    above = obj + np.array([0.0, 0.0, 0.55])
    at = obj + np.array([0.0, 0.0, 0.38])
    place = np.array([rng.uniform(-0.3, -0.15), rng.uniform(0.15, 0.30), 0.02])

    # TODO: replace the hardcoded joint targets with _ik_for_offset(). A 2-link
    # closed-form IK for Joint_1/Joint_2 has not been written; these constants
    # are placeholders that do not actually align the jaws with the object.
    plan = {
        "APPROACH":  (above, [0.9, 0.5], GRIPPER_OPEN),
        "DESCEND":   (at,    [1.2, 0.9], GRIPPER_OPEN),
        "GRASP":     (at,    [1.2, 0.9], GRIPPER_CLOSED),
        "LIFT":      (above, [0.9, 0.5], GRIPPER_CLOSED),
        "TRANSPORT": (place + np.array([0, 0, 0.55]), [0.9, 0.5], GRIPPER_CLOSED),
        "PLACE":     (place + np.array([0, 0, 0.38]), [1.2, 0.9], GRIPPER_CLOSED),
        "RELEASE":   (place + np.array([0, 0, 0.38]), [1.2, 0.9], GRIPPER_OPEN),
    }

    frames = []
    frames_per_phase = int(1.0 * FPS)          # 1 s of wall-clock per phase
    # Physics runs at model timestep (1 ms); a frame is one 1/FPS interval, so
    # each recorded frame must advance the sim by 1/FPS of PHYSICS, not by a
    # single mj_step. Stepping once per frame would record 210 frames spanning
    # 0.21 s while declaring fps=30 (i.e. 7 s) -- every timestamp wrong by 33x,
    # and the arm would barely move within an episode.
    #
    # 1 ms does not divide 1/30 s evenly, so substeps rounds 33.33 -> 33 and an
    # episode spans 6.93 s of sim rather than 7.00 (a uniform 1% rate error).
    # Harmless: LeRobot derives timestamps as frame_index/fps, so the dataset is
    # internally consistent; only sim-time and declared-time differ. Set FPS=25
    # (40 substeps) or timestep=1/30000 if you ever need them to agree exactly.
    substeps = max(1, round((1.0 / FPS) / model.opt.timestep))

    for phase in PHASES:
        drone_xyz, joints, grip = plan[phase]
        controller.set_targets(drone_xyz, joints, grip)

        for _ in range(frames_per_phase):
            for _ in range(substeps):
                controller.step()
                mujoco.mj_step(model, data)

            frame = {
                "observation.state": get_state(model, data),
                "action": controller.action(),
                "task": task_text,
            }
            for key, cam in CAMERAS.items():
                renderer.update_scene(data, camera=cam)
                frame[f"observation.images.{key}"] = renderer.render().copy()
            frames.append(frame)

    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out_repo_id", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=str,
                    default="pick up the red cube and place it to the side")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    controller = ScriptedController(model, data)

    # "names" is required on video features: feature_utils.py does names[2]
    # with no None guard, so omitting it raises TypeError at policy construction
    # -- and that same check is what drives the HWC->CHW shape flip.
    # timestamp/frame_index/episode_index/index/task_index are auto-populated;
    # passing them is rejected as extra features.
    features = {
        f"observation.images.{k}": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
        for k in CAMERAS
    }
    features["observation.state"] = {
        "dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES,
    }

    dataset = LeRobotDataset.create(
        repo_id=args.out_repo_id, fps=FPS, features=features, robot_type="skygrip",
    )

    for ep in range(args.episodes):
        for f in run_episode(model, data, renderer, controller, rng, args.task):
            dataset.add_frame(f)          # v3.0: task lives INSIDE the frame dict
        dataset.save_episode()
        print(f"episode {ep + 1}/{args.episodes} ({len(PHASES) * FPS} frames)")

    # Without finalize() the parquet footer is never written and the dataset
    # is invalid -- not merely incomplete.
    dataset.finalize()
    print("done. dataset.push_to_hub() to upload.")


if __name__ == "__main__":
    main()
