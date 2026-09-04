"""Multi-episode ground-truth replay: expert episodes 0..N-1 on
V2Runner(97000) captured, then replayed through the FIXED V2Platform
(proximity deploy + collector-mirrored weld gate) on a twin runner.
Validates both objects' aperture windows and the between-episode
FF cleanup."""
import sys
import numpy as np
import mujoco

sys.path.insert(0, r"c:/Users/hanah/Projects/drone-version2/Sim'n'Real/Mujoco")
import collect_airvla as A
import collect_v2 as V

SEED = 97000
N_EP = 3


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


class V2Platform:
    """FIXED platform (matches eval_v2.py post-fix)."""

    def __init__(self, rr, start, yaw0):
        self.r = rr
        self.sp = np.array(start, dtype=float)
        self.yaw = float(yaw0)
        self.grip = 1.0
        self.deployed = False
        self.near = 0
        self.cmd = np.array(rr.expert.q_travel, dtype=float)
        self.i = 0
        self.weld_tick = None
        self.released = False

    def tick(self, act, nav=False):
        rr = self.r
        self.sp = self.sp + np.clip(act[0:3], -0.035, 0.035)
        self.yaw += float(np.clip(act[5], -0.06, 0.06))
        self.grip = float(np.clip(act[6], 0.0, 1.0))
        welded = bool(rr.data.eq_active[rr.weld])
        ap = float(rr.data.qpos[rr.gadr])
        if not nav and not welded:
            aim2, _ = rr.live_target()
            pc = rr.jaws() - aim2
            if (float(np.linalg.norm(pc[0:2])) < 0.010
                    and abs(float(pc[2])) < 0.015
                    and rr.cur["ap_lo"] < ap * 1000 < rr.cur["ap_hi"]):
                rr.weld_grasp(True)
        elif welded and self.grip > 0.8:
            rr.weld_grasp(False)
            self.released = True
        welded = bool(rr.data.eq_active[rr.weld])
        if welded and self.weld_tick is None:
            self.weld_tick = self.i
        if not self.deployed and not nav:
            aim3, _ = rr.live_target()
            dxy = float(np.linalg.norm(
                np.asarray(rr.data.qpos[0:2]) - np.asarray(aim3[0:2])))
            self.near = self.near + 1 if dxy < 0.60 else 0
            if self.near >= 5:
                self.deployed = True
        tgt = (rr.expert.q_carry if (self.deployed and not nav
                                     and not self.released)
               else rr.expert.q_travel)
        self.cmd = self.cmd + np.clip(np.array(tgt) - self.cmd,
                                      -0.06, 0.06)
        self.i += 1
        rr.ctrl.set_targets(self.sp, self.cmd,
                            A.C.GRIPPER_OPEN * self.grip)
        rr.ctrl.mppi.target_yaw = self.yaw
        for _ in range(rr.sub):
            rr.ctrl.step()
            mujoco.mj_step(rr.model, rr.data)


r = V.V2Runner(SEED)
r2 = V.V2Runner(SEED)

for ep in range(N_EP):
    acts.clear()
    frames, info = r.episode_v2()
    ep_acts = [a.copy() for a in acts]
    exp_weld = info["grasped"]

    obj2, start2, alt2, tgt2 = r2.reset_scene_v2()
    plat = V2Platform(r2, start2, 0.0)
    adr2 = r2.cur["adr"]
    z0 = float(r2.data.qpos[adr2 + 2])
    miss = 1e9
    rise = -1e9
    lifted = False
    for a in ep_acts:
        plat.tick(a)
        aim, _ = r2.live_target()
        miss = min(miss, float(np.linalg.norm(r2.jaws() - aim)))
        oz = float(r2.data.qpos[adr2 + 2])
        rise = max(rise, oz - z0)
        if oz > A.MAT_TOP + 0.12 and bool(r2.data.eq_active[r2.weld]):
            lifted = True
    ended_welded = bool(r2.data.eq_active[r2.weld])
    if r2._ff_on:
        r2.weld_grasp(False)
    print("EP%d obj=%-14s expert_grasped=%-5s | replay weld_tick=%s "
          "picked(>12cm)=%-5s miss=%.1fmm rise=%+.0fmm ended_welded=%s"
          % (ep, info["obj"], exp_weld, plat.weld_tick, lifted,
             miss * 1000, rise * 1000, ended_welded), flush=True)

print("DONE", flush=True)
