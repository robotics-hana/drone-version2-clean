"""Bare scene, drone holding a stationary hover under the PD flight controller.

Same viewer as before, but launched passive so we can step the physics ourselves
and run the controller in the loop -- launch(model, data) alone just drops the
drone, because nothing is producing thrust.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from pd_flight import PDFlightController

ctrl = PDFlightController("SkyGrip_core.xml")
model, data = ctrl.model, ctrl.data

# Start AT the hover setpoint so there is no initial dive.
target = np.array([0.0, 0.0, 1.2])
data.qpos[0:3] = target
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
ctrl.target_pos = target.copy()

u = np.zeros(6)
u[0] = ctrl.nominal_hover_thrust
last_plan = -np.inf

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        # Physics at 1 kHz, replanning at the controller's 20 Hz -- same split
        # collect_demos.py uses.
        if data.time - last_plan >= ctrl.CTRL_DT:
            u = ctrl.mppi_step(ctrl.get_state())
            last_plan = data.time
        ctrl.apply_control(u)
        mujoco.mj_step(model, data)
        viewer.sync()
        # Real-time pacing.
        leftover = model.opt.timestep - (time.time() - step_start)
        if leftover > 0:
            time.sleep(leftover)