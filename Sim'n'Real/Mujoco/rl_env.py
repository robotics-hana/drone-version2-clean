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

Observation (AMENDED 2026-09-08 -- the original 12-dim world-frame
obs made the teacher unlearnable: the creep law steers FROM THE
SETPOINT, fires and closes by OBJECT-SPECIFIC grasp parameters, and
world-frame components bake the approach heading into the data;
none of the three was observable, so a BC clone regressed to the
dataset-mean action and diverged closed-loop). Now 18 dims, planar
components in the BODY frame (fwd = heading, v2 convention yaw 0
faces -y): [jaw-to-aim body (3), SETPOINT-to-jaw body (3), velocity
body (3), angular velocity (3), aperture (1), yaw error to aim (1),
grip (1), grasp params ap_lo/ap_hi/close (3)]. The setpoint and the
grasp params are the controller's OWN state and task spec -- the
scripted servo uses exactly the same quantities, so this is no new
privilege; the jaw-to-aim vector remains the documented
training-time sim privilege.

Action: residual 4-vector [d_fwd, d_left, dz, dgrip] in the BODY
frame (rotated to world inside step), bounded |dxyz| <= 0.01 m/tick,
dgrip in [-0.2, 0.2] added to the base grip; yaw residual
deliberately excluded (measured heading error is small).

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


def markov_tick(plat, rr):
    """One tick of the Markovian terminal teacher (BC/DAgger
    labeler): platform_v2._servo_tick's law with the approach axis
    re-derived from the current bearing each tick (so the action is
    a function of the observation) and no forward ratchet. Inside
    8 mm the bearing direction is numerically unstable, so the body
    heading substitutes (v2 convention: yaw 0 faces -y). Lives here,
    not in rl_bc, so diagnostics can import it without executing the
    BC script."""
    aim, _ = rr.live_target()
    u = aim[0:2] - rr.jaws()[0:2]
    d = float(np.linalg.norm(u))
    if d > 0.008:
        ufix = u / max(1e-9, d)
    else:
        ufix = np.array([np.sin(plat.yaw), -np.cos(plat.yaw)])
    plat.a_zc = 0.85 * plat.a_zc + 0.15 * float((rr.jaws() - aim)[2])
    cy, sy = np.cos(plat.yaw), np.sin(plat.yaw)
    oc = rr.expert.off_carry
    offw = np.array([cy * oc[0] - sy * oc[1],
                     sy * oc[0] + cy * oc[1], oc[2]])
    goal = aim - offw
    goal[2] -= min(plat.a_zc, 0.0)
    goal[0:2] += 0.012 * ufix
    dsp = goal - plat.sp
    n = float(np.linalg.norm(dsp))
    plat.sp = plat.sp + (dsp if n < 0.003 else dsp / n * 0.003)
    if not plat.a_fired:
        win_mid = (rr.cur["ap_lo"] + rr.cur["ap_hi"]) / 2.0
        ramp = max(2.0, (16.0 - win_mid) / 0.8)
        if d < 0.004 + 0.003 * ramp:
            plat.a_fired = True
    plat.grip = float(rr.cur["close"]) if plat.a_fired else 1.0


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

    def _frame(self):
        """Body-frame basis (fwd, left) for the current heading
        (v2 convention: yaw 0 faces -y)."""
        y = self.plat.yaw
        fwd = np.array([np.sin(y), -np.cos(y)])
        left = np.array([np.cos(y), np.sin(y)])
        return fwd, left

    def _to_body(self, v):
        fwd, left = self._frame()
        return np.array([v[0:2] @ fwd, v[0:2] @ left, v[2]])

    def _to_world(self, v):
        fwd, left = self._frame()
        xy = v[0] * fwd + v[1] * left
        return np.array([xy[0], xy[1], v[2]])

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
        return np.concatenate([
            self._to_body(aim - jaw),
            self._to_body(np.asarray(self.plat.sp, dtype=float) - jaw),
            self._to_body(vel), ang,
            [ap, yerr, self.plat.grip],
            [float(r.cur["ap_lo"]), float(r.cur["ap_hi"]),
             float(r.cur["close"])]]).astype(np.float32)

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
        """residual: BODY-frame [d_fwd, d_left, dz, dgrip]; base_act:
        full world-frame 7-vector or None (zero base = hover, grip
        held at its current value so dgrip acts as a rate)."""
        r = self.r
        res = np.asarray(residual, dtype=float)
        d_xyz = self._to_world(np.clip(res[0:3], -DXYZ_MAX, DXYZ_MAX))
        d_grip = float(np.clip(res[3], -DGRIP_MAX, DGRIP_MAX))
        if base_act is None:
            # grip channel is ABSOLUTE in this action space, so the
            # base must carry the CURRENT grip for dgrip to act as a
            # rate; the original 1.0 (open) base clamped the command
            # to >=0.8 and made the weld unreachable by construction
            # (found 2026-09-08: BC clone welded 1.00 in collection
            # with base[6]=plat.grip but 0/8 closed-loop on the
            # default base -- and retro-explains the 1,092-episode
            # zero-weld "exploration cliff", which was not
            # exploration at all)
            base_act = np.array([0, 0, 0, 0, 0, 0, self.plat.grip])
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
