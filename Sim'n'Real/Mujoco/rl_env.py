"""rl_env.py -- E5 terminal-phase residual-RL environment
(pre-registered 2026-09-07).

Each episode: the EXPERT flies the standard approach to the deployed
standoff (pick_v2(approach_only=True) -- the proven choreography),
then control passes to a V2Platform driven by base actions plus the
residual under training. Episodes are therefore short (~200 ticks)
and every sample sits in the decision-relevant terminal region.

Base-action provider is pluggable:
  - "zero": hover-in-place base (scaffolding smoke, and a meaningful
    ablation: can the residual alone learn the terminal leg?);
  - a callable (obs_dict -> 7-vector) for the pi0 base on the
    cluster (chunked at the naive cadence by the caller's wrapper).

Observation (proprio + sim-privileged target vector, documented as
training-time privilege): [jaw-to-aim vector (3), body velocity (3),
body angular velocity (3), aperture (1), yaw error to aim (1),
grip (1)] = 12 dims.

Action: residual 4-vector [dxyz (3), dgrip (1)], bounded |dxyz| <=
0.01 m/tick, dgrip in [-0.2, 0.2] added to the base grip; yaw
residual deliberately excluded (measured heading error is small).

Reward per tick: -||jaw-to-aim|| (metres)
  + 10.0 on weld (terminal)
  - 0.5 per table-contact tick, -0.5 per object-strike tick
  - 0.01 * ||residual||^2
  - 0.002 per tick (time pressure)
Timeout 300 ticks (terminal, no bonus).
"""
import numpy as np

import collect_airvla as A
import collect_v2 as V
from platform_v2 import V2Platform

MAX_TICKS = 300
DXYZ_MAX = 0.01
DGRIP_MAX = 0.2


class TerminalEnv:

    def __init__(self, seed, base="zero"):
        self.r = V.V2Runner(seed)
        self.base = base
        self.plat = None
        self.t = 0
        self._prev_hits = (0, 0)

    # -- helpers ----------------------------------------------------------
    def _aim(self):
        aim, _ = self.r.live_target()
        return np.asarray(aim, dtype=float)

    def obs(self):
        r = self.r
        aim = self._aim()
        jaw = np.asarray(r.jaws(), dtype=float)
        vel = np.asarray(r.data.qvel[0:3], dtype=float)
        ang = np.asarray(r.data.qvel[3:6], dtype=float)
        ap = float(r.data.qpos[r.gadr])
        # yaw error toward the aim (v2 heading convention: 0 faces -y)
        des = float(np.arctan2(aim[0] - r.data.qpos[0],
                               -(aim[1] - r.data.qpos[1])))
        yerr = (self.plat.yaw - des + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([aim - jaw, vel, ang,
                               [ap, yerr, self.plat.grip]]).astype(
                                   np.float32)

    def reset(self):
        r = self.r
        obj, start, alt, tgt = r.reset_scene_v2()
        task = A.PROMPT_MANIP.format(obj=obj)
        # expert flies the approach; frames are discarded (no vision
        # in the residual's observation)
        frames = []
        r.pick_v2(frames, task, alt, approach_only=True)
        # platform takes over exactly where the expert stopped: its
        # setpoint continues from the expert's, heading preserved
        self.plat = V2Platform(r, np.asarray(r.expert.sp, dtype=float),
                               float(r.expert.yaw))
        self.plat.deployed = True       # the expert just deployed it
        self.t = 0
        self._prev_hits = (r._table_hits, r._obj_hits)
        return self.obs()

    def step(self, residual, base_act=None):
        """residual: [dx, dy, dz, dgrip]; base_act: full 7-vector or
        None (zero base = hover + open grip drift toward 1)."""
        r = self.r
        res = np.asarray(residual, dtype=float)
        d_xyz = np.clip(res[0:3], -DXYZ_MAX, DXYZ_MAX)
        d_grip = float(np.clip(res[3], -DGRIP_MAX, DGRIP_MAX))
        if base_act is None:
            base_act = np.array([0, 0, 0, 0, 0, 0, 1.0])
        act = np.asarray(base_act, dtype=float).copy()
        act[0:3] = np.clip(act[0:3] + d_xyz, -0.035, 0.035)
        act[6] = float(np.clip(act[6] + d_grip, 0.0, 1.0))
        pre_d = float(np.linalg.norm(self._aim() - r.jaws()))
        self.plat.tick(act)
        self.t += 1
        aim = self._aim()
        d = float(np.linalg.norm(aim - r.jaws()))
        welded = bool(r.data.eq_active[r.weld])
        th, oh = r._table_hits, r._obj_hits
        new_table = th - self._prev_hits[0]
        new_obj = oh - self._prev_hits[1]
        self._prev_hits = (th, oh)
        reward = (-d
                  - 0.5 * new_table - 0.5 * new_obj
                  - 0.01 * float(res @ res)
                  - 0.002)
        done = False
        if welded:
            reward += 10.0
            done = True
        elif self.t >= MAX_TICKS:
            done = True
        info = dict(d=d, pre_d=pre_d, welded=welded, t=self.t,
                    table=new_table, obj=new_obj)
        if done and r._ff_on:           # trim-leak guard between eps
            r.weld_grasp(False)
        return self.obs(), reward, done, info


if __name__ == "__main__":
    # scaffolding smoke: zero-residual and random-residual rollouts,
    # render-free
    import sys
    V.V2Runner.frame = lambda self, task: {
        "observation.state": None, "scene_state": None,
        "action": None, "task": task}
    env = TerminalEnv(int(sys.argv[1]) if len(sys.argv) > 1 else 55000)
    rng = np.random.default_rng(0)
    for mode in ("zero", "random"):
        for ep in range(2):
            o = env.reset()
            total = 0.0
            info = {}
            done = False
            while not done:
                res = (np.zeros(4) if mode == "zero"
                       else rng.uniform(-1, 1, 4)
                       * [DXYZ_MAX] * 3 + [0])
                o, rew, done, info = env.step(res)
                total += rew
            print("SMOKE %-6s ep%d: ticks=%d final_d=%.0fmm "
                  "welded=%s return=%.2f"
                  % (mode, ep, info["t"], info["d"] * 1000,
                     info["welded"], total), flush=True)
    print("RLENV-SMOKE-DONE", flush=True)
