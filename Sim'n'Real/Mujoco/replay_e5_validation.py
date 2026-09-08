"""replay_e5_validation.py -- ground-truth validation of the E5
learned-servo handoff (platform_e5.V2PlatformLearned with the frozen
e5_actor_final.pt), mirroring replay_h1_validation case-for-case.

Cases (V2Runner(97000) scenes, same expert stream as H1's):
  A. assist_r=0, expert actions: must reproduce the banked replay
     EXACTLY (weld ticks 292/195/241) -- proves the learned subclass
     is inert when assist is off (only _servo_tick is overridden and
     it can never be reached).
  B. assist_r=0.15, expert actions: engagement near the object must
     not break the expert's grasp.
  C. assist_r=0.15, STALLING pilot (freeze inside the radius at
     0.13 m): the LEARNED servo must engage and complete the weld
     alone -- the test that makes the E5 rung meaningful.
  N. stall at 0.25 m (outside): must not engage.
"""
import sys

import numpy as np

sys.path.insert(0, r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco")
import collect_airvla as A
import collect_v2 as V
from platform_e5 import V2PlatformLearned

SEED = 97000
N_EP = 3
ACTOR = (sys.argv[1] if len(sys.argv) > 1 else
         r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco/"
         r"e5_actor/e5_actor_final.pt")
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
    plat = V2PlatformLearned(runner, start, 0.0, assist_r=assist_r,
                             actor_path=ACTOR)
    stalled = False
    for a in ep_acts:
        if stall_at is not None and not stalled:
            aim, _ = runner.live_target()
            if float(np.linalg.norm(
                    runner.jaws()[0:2] - aim[0:2])) < stall_at:
                stalled = True
        plat.tick(STALL_ACT if stalled else a)
    print("%s obj=%-14s weld_tick=%s assisted=%s assist_ticks=%d"
          % (label, obj, plat.weld_tick,
             plat.took_tick is not None, plat.assist_ticks),
          flush=True)
    if plat.r._ff_on:
        plat.r.weld_grasp(False)
    return plat.weld_tick


expected = [292, 195, 241]
ok = True
rCapA = V.V2Runner(SEED)
rA = V.V2Runner(SEED)
rB = V.V2Runner(SEED)
for ep in range(N_EP):
    acts.clear()
    frames, info = rCapA.episode_v2()
    ep_acts = [a.copy() for a in acts]
    wa = run_case("A ep%d" % ep, ep_acts, rA, 0.0)
    if wa != expected[ep]:
        ok = False
        print("  CASE A REGRESSION: expected weld %s" % expected[ep])
    wb = run_case("B ep%d" % ep, ep_acts, rB, 0.15)
    if wb is None:
        ok = False
        print("  CASE B FAILED: no weld with assist")
print("--- case C (stalling pilot, learned servo rescues) ---")
rCapC = V.V2Runner(SEED)
rC = V.V2Runner(SEED)
for ep in range(N_EP):
    acts.clear()
    frames, info = rCapC.episode_v2()
    ep_acts = [a.copy() for a in acts]
    wc = run_case("C ep%d" % ep, ep_acts, rC, 0.15, stall_at=0.13)
    if wc is None:
        ok = False
        print("  CASE C FAILED: learned servo did not rescue")
print("--- negative control: stall OUTSIDE the radius ---")
rCapN = V.V2Runner(SEED)
rN = V.V2Runner(SEED)
acts.clear()
frames, info = rCapN.episode_v2()
ep_acts = [a.copy() for a in acts]
wn = run_case("N ep0", ep_acts, rN, 0.15, stall_at=0.25)
if wn is not None:
    ok = False
    print("  NEGATIVE CONTROL FAILED: engaged beyond radius")
print("E5-VALIDATION", "PASS" if ok else "FAIL")
