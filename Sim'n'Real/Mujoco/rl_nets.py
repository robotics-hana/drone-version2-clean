"""rl_nets.py -- shared E5 classes, import-safe (no executable body).
Split out 2026-09-07: rl_bc importing rl_train executed the whole
PPO trainer at import time (rl_train had no __main__ guard); the
classes now live here so both scripts import without side effects."""
import torch
import torch.nn as nn


class _NoRenderer:
    """GPU-less-node stub: E5 never renders, but V2Runner's
    constructor builds a mujoco.Renderer, whose EGL context cannot
    exist on a CPU node."""

    def __init__(self, *a, **k):
        pass

    def update_scene(self, *a, **k):
        pass

    def render(self, *a, **k):
        return None

    def close(self):
        pass


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=18):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.mu = nn.Linear(128, 4)
        self.logstd = nn.Parameter(torch.full((4,), -0.7))
        self.v = nn.Linear(128, 1)

    def dist(self, obs):
        h = self.body(obs)
        return torch.distributions.Normal(
            torch.tanh(self.mu(h)), self.logstd.exp()), self.v(h)
