"""rl_bc.py -- E5 stage 0: DAgger-clone the Markovian terminal
servo into the RL actor, so PPO starts from a policy that already
welds.

History of this stage (all ledgered in finaldroneresults.md):
- Amendment 2026-09-07: 1,092 random-exploration episodes found zero
  welds, so a BC warm start was added. Post-mortem 2026-09-08: that
  was NOT a hard-exploration cliff -- rl_env's zero base held the
  absolute grip channel at 1.0 (open) and the |dgrip| <= 0.2 residual
  could never reach a closed grip, so the weld was unreachable by
  construction (fixed in rl_env: base grip = current grip).
- Amendment 2026-09-08 (gates 1-3, each 0/8 closed-loop): a plain
  clone of the servo fails for stacked reasons, found one per gate:
  (1) the servo's frozen approach axis + forward ratchet are internal
  latches not in the obs (teacher not a function of obs) -> Markovian
  teacher (rl_env.markov_tick, axis from current bearing, validated
  6/6); (2) the setpoint the creep law steers from, the
  object-specific grasp params, and the world-frame components were
  still unobservable -> 18-dim body-frame obs (rl_env docstring) +
  DART noise injection for tube coverage; (3) traced closed-loop:
  the clone approaches perfectly (379->42 mm) but NEVER STOPS -- the
  stop/fire transition is ~2-5% of ticks and conflicts with smooth
  approach labels, so it flies through the object at 3 mm/tick (the
  measured 300-tick end distances, 256-800 mm, are exactly
  fly-through) -- terminal loss weighting alone did not fix it
  (wmse plateau 0.042).
- Therefore: DAGGER. Roll the CLONE, label every visited state with
  the Markovian teacher (it labels any state -- the reason it was
  built), aggregate, retrain. The fly-through states then enter the
  dataset carrying "reverse" labels and the in-ball states carry
  "close now" labels, which is exactly the signal uniform BC lacked.

usage: python rl_bc.py <out_pt> [--seed 56000] [--episodes 120]
                       [--rounds 3] [--evaln 6]
  --episodes: round-0 teacher (DART) rollouts; each DAgger round
  adds episodes//2 clone rollouts. --evaln closed-loop eval episodes
  after each round (the go/no-go signal in the log).
"""
import sys

import numpy as np
import torch

import collect_v2 as V
from rl_env import TerminalEnv, DXYZ_MAX, DGRIP_MAX, markov_tick
from rl_nets import ActorCritic, _NoRenderer

import mujoco as _mj
_mj.Renderer = _NoRenderer

OUT = sys.argv[1]


def arg(flag, default):
    return (int(sys.argv[sys.argv.index(flag) + 1])
            if flag in sys.argv else default)


SEED = arg("--seed", 56000)
EPISODES = arg("--episodes", 120)
ROUNDS = arg("--rounds", 3)
EVALN = arg("--evaln", 6)

V.V2Runner.frame = lambda self, task: {
    "observation.state": None, "scene_state": None,
    "action": None, "task": task}

# DART noise on round-0 teacher rollouts (tube coverage); DAgger
# rounds use the clone's own actions plus a smaller dither.
NOISE_XYZ = 0.12                        # in tanh units (of DXYZ_MAX)
NOISE_GRIP = 0.05
DAGGER_DITHER = 0.05

torch.manual_seed(SEED)


def teacher_label(env):
    """Clean Markovian-teacher action at the env's CURRENT state,
    body frame, tanh units.

    Two rules beyond markov_tick, both needed only for DAgger's
    off-trajectory states (found on gate 4, where the raw latch
    poisoned 63% of the dataset with far-from-object 'close' labels):
    - the fire latch is inferred from the grip (which is in the obs),
      not carried as hidden state: an open grip means un-fired, so a
      clone that has drifted far away is taught to approach open, not
      to clench shut where a weld is impossible;
    - recovery: a closed-but-empty grip far from the object (the
      aftermath of a missed fire) is labeled REOPEN -- the scripted
      teacher never needs this, a learner recovering from its own
      miss does."""
    plat = env.plat
    aim = env._aim()
    d_h = float(np.linalg.norm(aim[0:2] - env.r.jaws()[0:2]))
    plat.a_fired = plat.grip < 0.8
    sp0 = plat.sp.copy()
    grip0 = plat.grip
    markov_tick(plat, env.r)
    d_xyz = plat.sp - sp0
    d_grip = plat.grip - grip0
    plat.sp = sp0
    plat.grip = grip0
    welded = bool(env.r.data.eq_active[env.r.weld])
    if grip0 < 0.8 and not welded and d_h > 0.025:
        d_grip = DGRIP_MAX              # missed fire: reopen
    body_xyz = env._to_body(np.clip(d_xyz, -DXYZ_MAX, DXYZ_MAX))
    return np.concatenate([body_xyz / DXYZ_MAX,
                           [np.clip(d_grip, -DGRIP_MAX,
                                    DGRIP_MAX) / DGRIP_MAX]])


def to_env_act(a):
    return np.concatenate([np.clip(a[:3], -1, 1) * DXYZ_MAX,
                           [np.clip(a[3], -1, 1) * DGRIP_MAX]])


def mu_of(net, obs):
    with torch.no_grad():
        return torch.tanh(net.mu(net.body(
            torch.as_tensor(obs, dtype=torch.float32)))).numpy()


def rollout(env, rng, X, Y, driver, n_eps, tag):
    """Collect (obs, teacher label) pairs along trajectories driven
    by `driver` (None = noisy teacher; a net = noisy clone)."""
    welds = 0
    for ep in range(n_eps):
        obs = env.reset()
        done = False
        while not done:
            target = teacher_label(env)
            X.append(obs.copy())
            Y.append(target.astype(np.float32))
            if driver is None:
                a = target + rng.normal(
                    0, [NOISE_XYZ] * 3 + [NOISE_GRIP])
            else:
                a = mu_of(driver, obs) + rng.normal(0, DAGGER_DITHER, 4)
            obs, rew, done, info = env.step(to_env_act(a))
        welds += int(info["welded"])
    print("%s: %d eps, welds=%d, dataset=%d pairs"
          % (tag, n_eps, welds, len(X)), flush=True)
    return welds


def train(X, Y, seed):
    """Fresh weighted regression on the aggregate dataset. Terminal
    weighting kept (fire x20, near x5) -- necessary but, without
    DAgger states, not sufficient."""
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Y_t = torch.as_tensor(np.asarray(Y), dtype=torch.float32)
    d_obs = torch.linalg.norm(X_t[:, 0:3], dim=1)
    W = (1.0 + 19.0 * (Y_t[:, 3].abs() > 0.5).float()
         + 4.0 * (d_obs < 0.06).float())
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(X_t)
    idx = np.arange(n)
    sh = np.random.default_rng(seed)
    for epoch in range(30):
        sh.shuffle(idx)
        tot = 0.0
        for s in range(0, n, 512):
            b = idx[s:s + 512]
            mu = torch.tanh(net.mu(net.body(X_t[b])))
            per = ((mu - Y_t[b]) ** 2).mean(-1)
            loss = (W[b] * per).sum() / W[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
    print("  train: %d pairs (fire %d, near %d) wmse %.5f"
          % (n, int((Y_t[:, 3].abs() > 0.5).sum()),
             int((d_obs < 0.06).sum()), tot / n), flush=True)
    return net


def closed_loop_eval(net, seed, n_eps):
    env = TerminalEnv(seed)
    welds = 0
    for ep in range(n_eps):
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(to_env_act(mu_of(net, obs)))
        welds += int(info["welded"])
    return welds


env = TerminalEnv(SEED)
rng = np.random.default_rng(SEED)
X, Y = [], []
rollout(env, rng, X, Y, None, EPISODES, "R0 teacher+DART")
net = train(X, Y, SEED)
w = closed_loop_eval(net, SEED + 1000, EVALN)
print("ROUND 0 closed-loop welds %d/%d" % (w, EVALN), flush=True)
best, best_w = net, w
for rd in range(1, ROUNDS + 1):
    rollout(env, rng, X, Y, net, max(1, EPISODES // 2),
            "R%d clone" % rd)
    net = train(X, Y, SEED + rd)
    w = closed_loop_eval(net, SEED + 1000, EVALN)
    print("ROUND %d closed-loop welds %d/%d" % (rd, w, EVALN),
          flush=True)
    if w >= best_w:
        best, best_w = net, w

# start PPO with tighter exploration around the cloned behaviour
with torch.no_grad():
    best.logstd.fill_(-1.5)
torch.save(best.state_dict(), OUT)
print("RLBC-DONE best_welds=%d/%d pairs=%d" % (best_w, EVALN, len(X)),
      flush=True)
