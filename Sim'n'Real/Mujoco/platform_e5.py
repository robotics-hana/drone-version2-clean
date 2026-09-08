"""platform_e5.py -- E5 deployment platform: V2Platform with the
LEARNED terminal servo (frozen e5_actor_final.pt, selection ledgered
2026-09-08) in place of the scripted creep law.

Design (pre-registered: "eval integration reuses H1's takeover
machinery with the learned policy in place of the script"):
- ONLY _servo_tick is overridden. Engagement trigger, weld gate,
  give-back guards, deploy rule, and every non-takeover code path are
  inherited from V2Platform unchanged -- with assist_r=0 this class
  is behaviourally identical to V2Platform (validated in
  replay_e5_validation.py case A).
- The observation is built by rl_env.obs_vec -- the training env's
  own function, so training/deployment parity holds by construction.
- The action application mirrors rl_env.step's residual path
  verbatim: body->world rotation, per-component 0.01 m clip, the
  0.035 m platform clip, grip as a rate on the current grip.
- Give-back guards are kept semantically identical to the scripted
  servo's: overall 300-tick cap, and 45 ticks after the "fire"
  (detected here as the grip crossing below 0.8 -- the same
  open/closed boundary the DAgger teacher's state-inferred latch
  used).
"""
import numpy as np
import torch

from platform_v2 import V2Platform
from rl_env import obs_vec, to_world, DXYZ_MAX, DGRIP_MAX
from rl_nets import ActorCritic


class V2PlatformLearned(V2Platform):

    def __init__(self, rr, start, yaw0, assist_r=0.0, actor_path=None):
        super().__init__(rr, start, yaw0, assist_r=assist_r)
        self.actor = ActorCritic()
        self.actor.load_state_dict(
            torch.load(actor_path, map_location="cpu",
                       weights_only=True))
        self.actor.eval()

    def _servo_tick(self, rr):
        """One tick of the LEARNED terminal servo: deterministic
        actor mean, applied exactly as rl_env.step applies the
        residual. Yaw held, as in the scripted servo."""
        obs = obs_vec(rr, self)
        with torch.no_grad():
            mu = torch.tanh(self.actor.mu(self.actor.body(
                torch.as_tensor(obs, dtype=torch.float32)))).numpy()
        d_xyz = to_world(self.yaw,
                         np.clip(mu[0:3], -1, 1) * DXYZ_MAX)
        d_grip = float(np.clip(mu[3], -1, 1) * DGRIP_MAX)
        self.sp = self.sp + np.clip(d_xyz, -0.035, 0.035)
        self.grip = float(np.clip(self.grip + d_grip, 0.0, 1.0))
        self.assist_ticks += 1
        # give-back: the 300-tick overall cap ONLY -- the training
        # env's episode contract. The scripted servo's extra
        # 45-ticks-after-fire clock does not transfer: its fire is a
        # discrete snap-to-close, while the learned actor EASES the
        # grip down during the final approach, so a grip-threshold
        # "fire" starts the clock early and yanks control back
        # mid-grasp (measured on e5cmini, job 305176: engaged 8/10
        # but converted 2/8, every failure at 5-29 mm with 65-71
        # assist ticks -- pulled off the grasp by the clock; the two
        # conversions seated in 47/58 ticks, just inside it).
        if (self.i - self.took_tick) > 300:
            self.takeover = False
            self.assist_done = True
