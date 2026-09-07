"""rl_bc.py -- E5 stage 0: behaviour-clone a MARKOVIAN terminal
servo into the RL actor, so PPO starts from a policy that already
welds.

History of this stage (both events ledgered in finaldroneresults.md):
- Amendment 2026-09-07: 1,092 random-exploration episodes found zero
  welds, so a BC warm start was added. Post-mortem 2026-09-08: that
  was NOT a hard-exploration cliff -- rl_env's zero base held the
  absolute grip channel at 1.0 (open) and the |dgrip| <= 0.2 residual
  could never reach a closed grip, so the weld was unreachable by
  construction (fixed in rl_env: base grip = current grip).
- Amendment 2026-09-08: cloning platform_v2._servo_tick verbatim
  fails closed-loop (0/8 welds despite MSE 0.0033 and cos 0.94 to
  the teacher on-distribution). Cause: the servo creeps along an
  axis FROZEN at engagement (a_ufix) with a forward ratchet (a_fwd),
  neither of which is in the 12-dim observation -- the teacher is
  not a function of obs, so the clone regresses the unresolvable
  x-component to a smeared mean (measured mean |err|: x 0.106 vs
  y 0.002) and the ~1 mm/tick bias compounds into divergence. The
  teacher below is the same law made Markovian: axis re-derived from
  the CURRENT bearing each tick (body-heading fallback inside 8 mm,
  where the bearing direction is numerically unstable), no ratchet.
  Validated 6/6 welds in TerminalEnv before adoption (t=115-148).
  platform_v2.py itself is untouched (frozen-eval dependency).

Mechanics: run TerminalEnv episodes driven by the Markovian teacher;
each tick's setpoint/grip change is undone and re-applied through
env.step() as the equivalent bounded action, so the recorded
(obs, action) pairs live exactly in the RL action space and the
episode dynamics are identical to PPO rollouts. The actor's tanh-mu
head is regressed onto the normalized actions; the critic is left at
init (PPO relearns it).

usage: python rl_bc.py <out_pt> [--seed 56000] [--episodes 120]
"""
import sys

import numpy as np
import torch

import collect_v2 as V
from rl_env import TerminalEnv, DXYZ_MAX, DGRIP_MAX
from rl_nets import ActorCritic, _NoRenderer

import mujoco as _mj
_mj.Renderer = _NoRenderer

OUT = sys.argv[1]
SEED = (int(sys.argv[sys.argv.index("--seed") + 1])
        if "--seed" in sys.argv else 56000)
EPISODES = (int(sys.argv[sys.argv.index("--episodes") + 1])
            if "--episodes" in sys.argv else 120)

V.V2Runner.frame = lambda self, task: {
    "observation.state": None, "scene_state": None,
    "action": None, "task": task}


def markov_tick(plat, rr):
    """One tick of the Markovian terminal teacher: _servo_tick's law
    with the approach axis re-derived from the current bearing (so
    the action is a function of the observation) and no forward
    ratchet. Inside 8 mm the bearing direction is unstable, so the
    body heading substitutes (v2 convention: yaw 0 faces -y)."""
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


env = TerminalEnv(SEED)
X, Y = [], []
welds = 0
for ep in range(EPISODES):
    obs = env.reset()
    plat = env.plat
    done = False
    while not done:
        sp0 = plat.sp.copy()
        grip0 = plat.grip
        markov_tick(plat, env.r)
        d_xyz = plat.sp - sp0
        d_grip = plat.grip - grip0
        # undo -- env.step() re-applies through the bounded path
        plat.sp = sp0
        plat.grip = grip0
        act = np.concatenate([np.clip(d_xyz, -DXYZ_MAX, DXYZ_MAX),
                              [np.clip(d_grip, -DGRIP_MAX,
                                       DGRIP_MAX)]])
        # normalized to actor output units (tanh range)
        target = np.concatenate([act[0:3] / DXYZ_MAX,
                                 [act[3] / DGRIP_MAX]])
        X.append(obs.copy())
        Y.append(target.astype(np.float32))
        obs, rew, done, info = env.step(act)
    welds += int(info["welded"])
    if (ep + 1) % 20 == 0:
        print("BC rollouts %d/%d welds=%d" % (ep + 1, EPISODES,
                                              welds), flush=True)
print("BC dataset: %d pairs, teacher weld rate %.2f"
      % (len(X), welds / EPISODES), flush=True)

X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
Y_t = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
n = len(X_t)
idx = np.arange(n)
for epoch in range(30):
    np.random.shuffle(idx)
    tot = 0.0
    for s in range(0, n, 512):
        b = idx[s:s + 512]
        mu = torch.tanh(net.mu(net.body(X_t[b])))
        loss = ((mu - Y_t[b]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    if (epoch + 1) % 10 == 0:
        print("BC epoch %d mse %.5f" % (epoch + 1, tot / n),
              flush=True)
# start PPO with tighter exploration around the cloned behaviour
with torch.no_grad():
    net.logstd.fill_(-1.5)
torch.save(net.state_dict(), OUT)
print("RLBC-DONE welds=%.2f pairs=%d" % (welds / EPISODES, len(X)),
      flush=True)
