"""rl_bc.py -- E5 stage 0: behaviour-clone the SCRIPTED terminal
servo into the RL actor, so PPO starts from a policy that already
welds (amendment 2026-09-07: 1,092 random-exploration episodes found
zero welds -- the weld event is unreachable by chance, the classic
hard-exploration cliff; BC-then-RL is the standard remedy and we own
a perfect demonstrator).

Mechanics: run episodes in TerminalEnv where the platform's own
_servo_tick computes each tick's sp/grip change; the change is undone
and re-applied through env.step() as the equivalent bounded action,
so the recorded (obs, action) pairs live exactly in the RL action
space. The actor's tanh-mu head is regressed onto the normalized
actions; the critic is left at init (PPO relearns it).

usage: python rl_bc.py <out_pt> [--seed 56000] [--episodes 120]
"""
import sys

import numpy as np
import torch
import torch.nn as nn

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

env = TerminalEnv(SEED)
X, Y = [], []
welds = 0
for ep in range(EPISODES):
    obs = env.reset()
    plat = env.plat
    # arm the servo machinery manually (no policy trigger here)
    plat.takeover = True
    plat.took_tick = 0
    done = False
    while not done:
        sp0 = plat.sp.copy()
        grip0 = plat.grip
        take0 = plat.takeover
        plat._servo_tick(env.r)
        d_xyz = plat.sp - sp0
        d_grip = plat.grip - grip0
        # undo -- env.step() re-applies through the bounded path
        plat.sp = sp0
        plat.grip = grip0
        plat.takeover = take0           # ignore give-back here
        plat.took_tick = 0              # and its timeout clock
        act = np.concatenate([np.clip(d_xyz, -DXYZ_MAX, DXYZ_MAX),
                              [np.clip(d_grip, -DGRIP_MAX,
                                       DGRIP_MAX)]])
        # normalized to actor output units (tanh range)
        target = np.concatenate([act[0:3] / DXYZ_MAX,
                                 [act[3] / DGRIP_MAX]])
        X.append(obs.copy())
        Y.append(target.astype(np.float32))
        # the platform must not double-apply: step with the action,
        # takeover OFF so tick() uses it as a plain policy action
        plat.takeover = False
        base = np.array([0, 0, 0, 0, 0, 0, 0.0])
        base[6] = plat.grip             # hold current grip as base
        obs, rew, done, info = env.step(act, base_act=base)
        plat.takeover = True
    welds += int(info["welded"])
    if (ep + 1) % 20 == 0:
        print("BC rollouts %d/%d welds=%d" % (ep + 1, EPISODES,
                                              welds), flush=True)
print("BC dataset: %d pairs, servo weld rate %.2f"
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
