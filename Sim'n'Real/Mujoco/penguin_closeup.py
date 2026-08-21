"""Close-up chase-camera video of penguin grasp episodes (user request):
a free camera tracks the gripper through approach, pinch, lift, carry
and drop, at 512 px, 10 Hz."""
import imageio.v2 as imageio
import mujoco
import numpy as np
import collect_airvla as A

r = A.Runner(52000)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 0.55
cam.elevation = -18
cam.azimuth = 205
rend = mujoco.Renderer(r.model, height=512, width=512)
clips = []
current = []

orig_step = A.Runner.step


def step_cu(self, frames, task):
    orig_step(self, frames, task)
    look = 0.5 * (self.jaws()
                  + self.data.qpos[self.cur["adr"]:self.cur["adr"] + 3])
    cam.lookat[:] = look
    rend.update_scene(self.data, camera=cam)
    current.append(rend.render().copy())


A.Runner.step = step_cu
kept = 0
tries = 0
while kept < 2 and tries < 6:
    tries += 1
    current.clear()
    _, info = r.manip_episode(obj="plush penguin")
    if info["success"] and info.get("gate", True):
        clips.append(list(current))
        kept += 1
        print(f"kept episode ({len(current)} frames)", flush=True)

w = imageio.get_writer("/clusterhome/hana/penguin_closeup.mp4", fps=10,
                       codec="libx264", quality=8, macro_block_size=1)
for clip in clips:
    for f in clip:
        w.append_data(f)
w.close()
print(f"WROTE closeup ({sum(len(c) for c in clips)} frames)", flush=True)
