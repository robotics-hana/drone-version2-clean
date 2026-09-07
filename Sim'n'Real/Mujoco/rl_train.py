"""rl_train.py -- E5: PPO training of the learned terminal controller
(pre-registered 2026-09-07; design refinement: zero-base = the agent
IS the terminal controller, no pi0 in the training loop).

Compact PPO against rl_env.TerminalEnv. No external RL dependency --
implemented directly on the pinned torch. CPU-friendly: the policy is
a 2x128 MLP over 12 obs dims; the simulator dominates wall time.

Actor-critic: tanh-Gaussian actor over 4 actions (scaled to the env
bounds inside the env), value head. Standard clipped-objective PPO
with GAE. Checkpoints + a scoreboard line (weld rate over the last
100 episodes) every iteration; stops at --episodes or --hours.

usage:
  python rl_train.py <out_dir> [--seed 55000] [--episodes 20000]
      [--hours 8] [--resume]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import collect_v2 as V
from rl_env import TerminalEnv, DXYZ_MAX, DGRIP_MAX

OUT = Path(sys.argv[1])


def arg(flag, default, cast=float):
    return (cast(sys.argv[sys.argv.index(flag) + 1])
            if flag in sys.argv else default)


SEED = int(arg("--seed", 55000))
MAX_EPISODES = int(arg("--episodes", 20000))
MAX_HOURS = arg("--hours", 8.0)
RESUME = "--resume" in sys.argv

# PPO hyperparameters (fixed, pre-registered; no tuning against eval)
ROLLOUT_TICKS = 4096
EPOCHS = 8
MINIBATCH = 256
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
LR = 3e-4
ENT = 0.001
VCOEF = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)

# render-free simulator
V.V2Runner.frame = lambda self, task: {
    "observation.state": None, "scene_state": None,
    "action": None, "task": task}


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(12, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.mu = nn.Linear(128, 4)
        self.logstd = nn.Parameter(torch.full((4,), -0.7))
        self.v = nn.Linear(128, 1)

    def dist(self, obs):
        h = self.body(obs)
        return torch.distributions.Normal(
            torch.tanh(self.mu(h)), self.logstd.exp()), self.v(h)


def act_scale(a):
    """Map the actor's ~[-1,1] output to env residual units."""
    a = np.asarray(a, dtype=float)
    return np.concatenate([a[0:3] * DXYZ_MAX, [a[3] * DGRIP_MAX]])


env = TerminalEnv(SEED)
net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=LR)
ep_count = 0
scoreboard = []                          # last-100 weld outcomes
state_path = OUT / "ppo_state.pt"
if RESUME and state_path.exists():
    st = torch.load(state_path, weights_only=False)
    net.load_state_dict(st["net"])
    opt.load_state_dict(st["opt"])
    ep_count = st["ep_count"]
    scoreboard = st["scoreboard"]
    print("RESUME at episode %d" % ep_count, flush=True)

t0 = time.time()
obs = env.reset()
ep_ret = 0.0
it = 0
while (ep_count < MAX_EPISODES
       and (time.time() - t0) / 3600 < MAX_HOURS):
    it += 1
    O, Ac, Lp, Rw, Dn, Vl = [], [], [], [], [], []
    for _ in range(ROLLOUT_TICKS):
        ot = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, val = net.dist(ot)
            a = dist.sample()
            lp = dist.log_prob(a).sum()
        nobs, rew, done, info = env.step(act_scale(a.numpy()))
        O.append(obs); Ac.append(a.numpy()); Lp.append(float(lp))
        Rw.append(rew); Dn.append(done); Vl.append(float(val))
        ep_ret += rew
        obs = nobs
        if done:
            ep_count += 1
            scoreboard.append(1 if info["welded"] else 0)
            scoreboard = scoreboard[-100:]
            obs = env.reset()
            ep_ret = 0.0
    # GAE
    with torch.no_grad():
        _, last_v = net.dist(torch.as_tensor(obs,
                                             dtype=torch.float32))
    adv = np.zeros(ROLLOUT_TICKS, dtype=np.float32)
    g = 0.0
    nv = float(last_v)
    for t in reversed(range(ROLLOUT_TICKS)):
        nonterm = 0.0 if Dn[t] else 1.0
        delta = Rw[t] + GAMMA * nv * nonterm - Vl[t]
        g = delta + GAMMA * LAM * nonterm * g
        adv[t] = g
        nv = Vl[t]
    ret = adv + np.asarray(Vl, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    O_t = torch.as_tensor(np.asarray(O), dtype=torch.float32)
    A_t = torch.as_tensor(np.asarray(Ac), dtype=torch.float32)
    L_t = torch.as_tensor(np.asarray(Lp), dtype=torch.float32)
    Ad_t = torch.as_tensor(adv)
    Rt_t = torch.as_tensor(ret)
    idx = np.arange(ROLLOUT_TICKS)
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, ROLLOUT_TICKS, MINIBATCH):
            b = idx[s:s + MINIBATCH]
            dist, val = net.dist(O_t[b])
            lp = dist.log_prob(A_t[b]).sum(-1)
            ratio = (lp - L_t[b]).exp()
            pg = -torch.min(
                ratio * Ad_t[b],
                ratio.clamp(1 - CLIP, 1 + CLIP) * Ad_t[b]).mean()
            vloss = ((val.squeeze(-1) - Rt_t[b]) ** 2).mean()
            ent = dist.entropy().sum(-1).mean()
            loss = pg + VCOEF * vloss - ENT * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    weld_rate = float(np.mean(scoreboard)) if scoreboard else 0.0
    print("ITER %d eps=%d weld100=%.2f pg=%.3f v=%.3f std=%s"
          % (it, ep_count, weld_rate, float(pg), float(vloss),
             np.round(net.logstd.exp().detach().numpy(), 3)),
          flush=True)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(),
                    ep_count=ep_count, scoreboard=scoreboard),
               state_path)
    if it % 10 == 0:
        torch.save(net.state_dict(), OUT / ("actor_it%04d.pt" % it))
    json.dump(dict(it=it, episodes=ep_count, weld100=weld_rate),
              open(OUT / "progress.json", "w"))
torch.save(net.state_dict(), OUT / "actor_final.pt")
print("RLTRAIN-DONE eps=%d weld100=%.2f"
      % (ep_count, float(np.mean(scoreboard)) if scoreboard else 0.0),
      flush=True)
