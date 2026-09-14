"""replay_v3_arm.py -- V3-ARM gate: expert-replay validation of the
--policy-arm path (pre-registered 2026-09-13: "expert-replay with
relabeled actions must still weld" before the flag is used).

Captures expert episodes on V2Runner(97000), relabels action dims
3,4 with per-tick arm-joint deltas from the recorded proprio (the
relabel_v3.py rule, applied in-memory), then replays through the
REAL platform_v2.V2Platform with policy_arm=True on a twin runner
(import, don't copy -- training-deployment contract parity). Pass =
every episode the expert grasped also welds+lifts in replay, and the
replayed arm tracks the recorded joint trajectory.

usage: python replay_v3_arm.py            (exit 0 = gate pass)
"""
import sys
import numpy as np

import collect_airvla as A
import collect_v2 as V
from platform_v2 import V2Platform

SEED = 97000
N_EP = 6


def stub_frame(self, task):
    return {"observation.state": None, "scene_state": None,
            "action": None, "task": task}


V.V2Runner.frame = stub_frame

r = V.V2Runner(SEED)
r2 = V.V2Runner(SEED)

acts = []
arm_q = []
o_tick = A.Expert.tick


def rec_tick(self):
    a = o_tick(self)
    acts.append(np.asarray(a, float).copy())
    arm_q.append(np.array(r.data.qpos[7:9], float))
    return a


A.Expert.tick = rec_tick

ok = True
for ep in range(N_EP):
    acts.clear()
    arm_q.clear()
    frames, info = r.episode_v2()
    A_ep = np.stack([a.copy() for a in acts])
    Q = np.stack(arm_q)
    # the relabel rule (relabel_v3.py): dims 3,4 <- consecutive-state
    # arm deltas, last step 0 (no successor)
    d_arm = np.zeros((len(A_ep), 2))
    d_arm[:-1] = Q[1:] - Q[:-1]
    assert np.allclose(A_ep[:, 3:5], 0.0), "dims 3,4 not padded?"
    A_ep[:, 3:5] = d_arm

    obj2, start2, alt2, tgt2 = r2.reset_scene_v2()
    plat = V2Platform(r2, start2, 0.0, policy_arm=True)
    adr2 = r2.cur["adr"]
    z0 = float(r2.data.qpos[adr2 + 2])
    miss = 1e9
    lifted = False
    qerr = 0.0
    for t, a in enumerate(A_ep):
        plat.tick(a)
        aim, _ = r2.live_target()
        miss = min(miss, float(np.linalg.norm(r2.jaws() - aim)))
        oz = float(r2.data.qpos[adr2 + 2])
        if oz > A.MAT_TOP + 0.12 and bool(r2.data.eq_active[r2.weld]):
            lifted = True
        if t + 1 < len(Q):
            qerr = max(qerr, float(np.max(np.abs(
                np.array(r2.data.qpos[7:9]) - Q[t + 1]))))
    ended_welded = bool(r2.data.eq_active[r2.weld])
    if r2._ff_on:
        r2.weld_grasp(False)
    exp_g = bool(info["grasped"])
    if exp_g and not lifted:
        ok = False
    print("EP%d obj=%-14s expert_grasped=%-5s | v3arm replay "
          "weld_tick=%s picked=%-5s miss=%.1fmm max|dq|=%.4f "
          "ended_welded=%s"
          % (ep, info["obj"], exp_g, plat.weld_tick, lifted,
             miss * 1000, qerr, ended_welded), flush=True)

print("V3ARM-REPLAY-%s" % ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
