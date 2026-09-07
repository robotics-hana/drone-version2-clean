"""replay_h1_validation.py -- ground-truth validation of the H1
terminal-servo handoff (and regression of the shared platform).

Three cases on the same expert episodes (V2Runner(97000) scenes):
  A. assist_r=0, expert actions: must reproduce the banked replay
     EXACTLY (weld ticks 292/195/241) -- proves the platform_v2
     refactor changed nothing for the pure-policy rungs.
  B. assist_r=0.15, expert actions: the servo may engage near the
     object; the grasp must still seat.
  C. assist_r=0.15, STALLING pilot: expert actions until the jaws
     come within 0.20 m of the aim, then all-zero actions (a policy
     that arrives and freezes -- the measured failure mode). The
     servo must engage and complete the weld ALONE. This is the
     test that makes the hybrid rung meaningful.
"""
import sys

import numpy as np

sys.path.insert(0, r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco")
import collect_airvla as A
import collect_v2 as V
from platform_v2 import V2Platform

SEED = 97000
N_EP = 3
STALL_ACT = np.array([0, 0, 0, 0, 0, 0, 1.0])   # hover, gripper open


def stub_frame(self, task):
    return {"observation.state": None, "scene_state": None,
            "action": None, "task": task}


V.V2Runner.frame = stub_frame

acts = []
o_tick = A.Expert.tick


def rec_tick(self):
    a = o_tick(self)
    acts.append(np.asarray(a, float).copy())
    return a


A.Expert.tick = rec_tick


def run_case(label, ep_acts, runner, assist_r, stall_at=None):
    obj, start, alt, tgt = runner.reset_scene_v2()
    plat = V2Platform(runner, start, 0.0, assist_r=assist_r)
    adr = runner.cur["adr"]
    z0 = float(runner.data.qpos[adr + 2])
    stalled = False
    rise = -1e9
    for a in ep_acts:
        if stall_at is not None and not stalled:
            aim, _ = runner.live_target()
            if float(np.linalg.norm(
                    runner.jaws()[0:2] - aim[0:2])) < stall_at:
                stalled = True
        plat.tick(STALL_ACT if stalled else a)
        rise = max(rise, float(runner.data.qpos[adr + 2]) - z0)
    print("%s obj=%-14s weld_tick=%s assisted=%s assist_ticks=%d "
          "rise=%+.0fmm" % (label, obj, plat.weld_tick,
                            plat.took_tick is not None,
                            plat.assist_ticks, rise * 1000),
          flush=True)
    if plat.r._ff_on:
        plat.r.weld_grasp(False)
    return plat.weld_tick


rA = V.V2Runner(SEED)      # capture + case A
rB = V.V2Runner(SEED)      # case B
rC = V.V2Runner(SEED)      # case C
expected = [292, 195, 241]
ok = True
for ep in range(N_EP):
    acts.clear()
    frames, info = rA.episode_v2()      # capture on rA's stream
    ep_acts = [a.copy() for a in acts]
    # case A: regression on a FRESH scene from rB? No -- A must use
    # the same scene; rA already consumed it for capture. Use rB for
    # case A (same seed stream => same scene), rC for B, and a 4th
    # runner for C would desync. Instead: rB hosts case A, rC hosts
    # case B, and case C re-uses rC? Each runner's stream must
    # advance once per episode; so create case C's runner lazily.
    wa = run_case("A ep%d" % ep, ep_acts, rB, 0.0)
    if wa != expected[ep]:
        ok = False
        print("  REGRESSION MISMATCH: expected weld %s" % expected[ep])
    wb = run_case("B ep%d" % ep, ep_acts, rC, 0.15)
    if wb is None:
        ok = False
        print("  CASE B FAILED: no weld with assist")
print("--- case C (stalling pilot) on a fresh stream ---")
rD = V.V2Runner(SEED)
for ep in range(N_EP):
    # replay the captured actions per episode on rD with stall
    # (re-capture to keep streams aligned)
    pass
# simpler: fresh capture runner + fresh case-C runner, same seed
rE = V.V2Runner(SEED)
rF = V.V2Runner(SEED)
for ep in range(N_EP):
    acts.clear()
    frames, info = rE.episode_v2()
    ep_acts = [a.copy() for a in acts]
    # stall INSIDE the assist radius (the measured policy failure
    # sits at 25-60 mm; the first test stalled at 0.20 m -- outside
    # 0.15 -- and correctly did not engage, which is the negative
    # control, kept below)
    wc = run_case("C ep%d" % ep, ep_acts, rF, 0.15, stall_at=0.13)
    if wc is None:
        ok = False
        print("  CASE C FAILED: servo did not rescue the stall")
print("--- negative control: stall OUTSIDE the radius ---")
rG = V.V2Runner(SEED)
rH = V.V2Runner(SEED)
acts.clear()
frames, info = rG.episode_v2()
ep_acts = [a.copy() for a in acts]
wn = run_case("N ep0", ep_acts, rH, 0.15, stall_at=0.25)
if wn is not None:
    ok = False
    print("  NEGATIVE CONTROL FAILED: servo engaged beyond radius")
print("H1-VALIDATION", "PASS" if ok else "FAIL")
