"""Open the drone inside the scanned lab, HOVERING under closed-loop control.

Same idea as t1.py, but on SkyGrip_lab.xml and actually flying: the PD flight
controller from pd_flight.py replans at its control rate inside the viewer loop,
so the drone holds station instead of drifting off on open-loop thrust (which is
what the first version of this script did -- open-loop hover is unstable by
definition, the thrust never exactly matches and there is nothing to pull the
body back).

    python t_lab.py            # the lab scene: hovers over the mat, bottle in view
    python t_lab.py --bare     # the original pedestal scene

In the viewer:
  * drag with right mouse / scroll to move the camera; the drone just hovers
  * press Tab for the left panel -> Rendering -> "Geom group": toggle group 2
    (the lab mesh) and group 3 (the invisible wall collision boxes)
  * the camera dropdown cycles scene_cam / overview_cam / wrist_cam -- what a
    policy would actually see
"""

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from pd_flight import PDFlightController

BARE = "--bare" in sys.argv
path = "SkyGrip_core.xml" if BARE else "SkyGrip_lab.xml"

ctrl = PDFlightController(path)
model, data = ctrl.model, ctrl.data

aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
jadr = lambda n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]

# Arm forward-and-down so the gripper is in shot, jaws open. qpos AND ctrl are
# set to the same pose before the first step, so the stiff position servos see
# zero error at t=0 and the arm does not slew (no reaction torque to fight).
ARM = (1.0, 0.7)
data.qpos[jadr("Joint_1")], data.qpos[jadr("Joint_2")] = ARM
data.qpos[jadr("right_clamp")], data.qpos[jadr("left_clamp")] = 0.016, -0.016

# Hover station. Lab: over the mat, a metre up, with the bottle and bin in front
# of the nose (the bottle is at (0, 0.6), the bin at (1.4, 1.6)). Bare: the
# usual spot above the pedestal.
target = np.array([0.0, 0.2, 1.0]) if not BARE else np.array([0.0, 0.0, 1.2])
data.qpos[0:3] = target                # start AT the setpoint -- no initial dive
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
ctrl.target_pos = target.copy()

print(f"model: {path}")
print(f"  all-up {ctrl.total_mass*1000:.0f} g, hover {ctrl.nominal_hover_thrust:.2f} N")
print(f"  holding station at {target} under the PD controller")
if not BARE:
    print("  mustard bottle at (0.0, 0.6), bin at (1.4, 1.6), mat centre (0.65, 1.0)")
print("launching viewer -- close the window to exit")

u = np.zeros(6)
u[0] = ctrl.nominal_hover_thrust
last_plan = -np.inf

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        # Replan at the controller's own rate, not every physics step -- the same
        # split collect_demos uses (physics 1 kHz, control 20 Hz).
        if data.time - last_plan >= ctrl.CTRL_DT:
            u = ctrl.mppi_step(ctrl.get_state())
            last_plan = data.time
        ctrl.apply_control(u)
        # Arm and gripper are position setpoints written after apply_control,
        # exactly as SkyGripController.step() does.
        data.ctrl[aid("act_joint1")], data.ctrl[aid("act_joint2")] = ARM
        data.ctrl[aid("act_gripper")] = 0.016
        mujoco.mj_step(model, data)
        viewer.sync()
        # Real-time pacing: sleep off whatever the step left of the timestep.
        leftover = model.opt.timestep - (time.time() - step_start)
        if leftover > 0:
            time.sleep(leftover)
