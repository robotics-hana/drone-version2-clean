"""platform_v2.py -- the arm/grasp automaton for v2 evaluation, plus
the optional H1 terminal-servo handoff ("hybrid" rung).

Extracted verbatim from eval_v2.py (2026-09-07) so the eval, the
replay validators, and the hybrid logic share ONE source. With
assist_r == 0 the behaviour is bit-identical to the previous inline
class (the pure-policy rungs are untouched).

History of the base class: ported from the v1 path, then REVALIDATED
BY EXPERT-ACTION REPLAY in the v2 world (2026-09-04), which falsified
two v1-heritage rules -- the arm deploys on sustained proximity to
the target (never on ascent), and the weld gate mirrors the
collector's verbatim (10/15 mm jaw-to-aim + per-object aperture
window, no pad-contact term).

H1 handoff (pre-registered 2026-09-07, Hana's directive): when the
POLICY brings the jaws within assist_r horizontally of the commanded
object (arm deployed, pre-weld), the platform's scripted terminal
servo -- the same creep law the expert uses, 600/600 + 310/310
validated in collection -- takes over ONLY the final leg: 3 mm/tick
setpoint creep along a frozen approach axis with +12 mm overshoot,
raise-only sag compensation, measured-rate fire distance, existing
weld gate. On weld (or timeout) control returns to the policy for
carry and place. The episode result records whether and when the
servo engaged, so hybrid numbers are always separable from
pure-policy numbers."""
import mujoco
import numpy as np

import collect_airvla as A


class V2Platform:

    def __init__(self, rr, start, yaw0, assist_r=0.0, policy_arm=False):
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
        # V3-ARM (pre-registered 2026-09-13): when True, action dims
        # 3,4 drive the arm joints directly as per-tick deltas
        # (clipped +-0.06/tick, the phase-slew bound), REPLACING the
        # q_travel/q_carry switching below. self.deployed still
        # updates (metrics only). Default False = frozen behaviour,
        # byte-identical path.
        self.policy_arm = bool(policy_arm)
        if self.policy_arm:
            m = rr.model
            lohi = [m.actuator_ctrlrange[mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)]
                    for nm in ("act_joint1", "act_joint2")]
            self._arm_lo = np.array([lohi[0][0], lohi[1][0]])
            self._arm_hi = np.array([lohi[0][1], lohi[1][1]])
        # H1 terminal-servo state (inert when assist_r == 0)
        self.assist_r = float(assist_r)
        self.takeover = False
        self.assist_done = False        # engage at most once
        self.took_tick = None
        self.assist_ticks = 0
        self.a_ufix = None              # frozen approach axis
        self.a_fwd = None               # forward ratchet
        self.a_zc = 0.0                 # filtered jaw-sag estimate
        self.a_fired = False
        self.a_fire_tick = None

    def _servo_tick(self, rr):
        """One tick of the scripted terminal servo (the expert's creep
        law, platform-side): drives self.sp/self.grip; yaw held."""
        aim, _ = rr.live_target()
        u = aim[0:2] - rr.jaws()[0:2]
        d = float(np.linalg.norm(u))
        if self.a_ufix is None:
            self.a_ufix = u / max(1e-9, d)
        # raise-only sag compensation (collector's filter constants)
        self.a_zc = 0.85 * self.a_zc + 0.15 * float(
            (rr.jaws() - aim)[2])
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        oc = rr.expert.off_carry
        offw = np.array([cy * oc[0] - sy * oc[1],
                         sy * oc[0] + cy * oc[1], oc[2]])
        goal = aim - offw
        goal[2] -= min(self.a_zc, 0.0)
        goal[0:2] += 0.012 * self.a_ufix
        ga = float(goal[0:2] @ self.a_ufix)
        if self.a_fwd is not None and ga < self.a_fwd:
            goal[0:2] += (self.a_fwd - ga) * self.a_ufix
        self.a_fwd = ga if self.a_fwd is None else max(self.a_fwd, ga)
        # creep the setpoint at the collector's 3 mm/tick
        dsp = goal - self.sp
        n = float(np.linalg.norm(dsp))
        self.sp = self.sp + (dsp if n < 0.003 else dsp / n * 0.003)
        # measured-finger-rate fire distance (collector's law)
        if not self.a_fired:
            win_mid = (rr.cur["ap_lo"] + rr.cur["ap_hi"]) / 2.0
            ramp = max(2.0, (16.0 - win_mid) / 0.8)
            if d < 0.004 + 0.003 * ramp:
                self.a_fired = True
                self.a_fire_tick = self.i
        self.grip = float(rr.cur["close"]) if self.a_fired else 1.0
        self.assist_ticks += 1
        # give-back guards: overall cap, and a bounded close (the
        # collector aborts at 30 ticks; here control just returns to
        # the policy -- the episode continues either way)
        if ((self.i - self.took_tick) > 300
                or (self.a_fired
                    and (self.i - self.a_fire_tick) > 45)):
            self.takeover = False
            self.assist_done = True

    def tick(self, act, nav=False):
        rr = self.r
        welded_pre = bool(rr.data.eq_active[rr.weld])
        # H1 trigger: policy brought the jaws near the commanded
        # object with the arm deployed -- hand the final leg to the
        # scripted servo (at most once per episode)
        if (self.assist_r > 0 and not nav and self.deployed
                and not self.released and not self.assist_done
                and not self.takeover and not welded_pre):
            aim, _ = rr.live_target()
            if float(np.linalg.norm(
                    rr.jaws()[0:2] - aim[0:2])) < self.assist_r:
                self.takeover = True
                self.took_tick = self.i
        if self.takeover and not welded_pre:
            self._servo_tick(rr)        # servo owns sp/grip this tick
        else:
            self.sp = self.sp + np.clip(act[0:3], -0.035, 0.035)
            self.yaw += float(np.clip(act[5], -0.06, 0.06))
            self.grip = float(np.clip(act[6], 0.0, 1.0))
        welded = bool(rr.data.eq_active[rr.weld])
        ap = float(rr.data.qpos[rr.gadr])
        # weld gate: the COLLECTOR's condition verbatim (see module
        # docstring; ground-truth-replay fix #2, 2026-09-04)
        if not nav and not welded:
            aim, _ = rr.live_target()
            pc = rr.jaws() - aim
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
        if welded and self.takeover:
            self.takeover = False       # grasp seated: policy resumes
            self.assist_done = True
        # deploy rule: sustained proximity (ground-truth-replay fix,
        # 2026-09-04 -- the expert deploys after settling at the
        # ~0.50 m standoff, never on ascent)
        if not self.deployed and not nav:
            aim, _ = rr.live_target()
            dxy = float(np.linalg.norm(
                np.asarray(rr.data.qpos[0:2]) - np.asarray(aim[0:2])))
            self.near = self.near + 1 if dxy < 0.60 else 0
            if self.near >= 5:
                self.deployed = True
        if self.policy_arm:
            self.cmd = np.clip(
                self.cmd + np.clip(np.asarray(act[3:5], dtype=float),
                                   -0.06, 0.06),
                self._arm_lo, self._arm_hi)
        else:
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
        # contact observations (collector cadence/dedupe; the
        # jaws-in-grasp-phase rule is deliberately omitted -- the
        # policy owns its grasp timing)
        d = rr.data
        table_hit = obj_hit = False
        for ci in range(d.ncon):
            g1 = int(d.contact.geom1[ci])
            g2 = int(d.contact.geom2[ci])
            if not table_hit and (
                    (g1 in rr._drone_geoms and g2 in rr._table_geoms)
                    or (g2 in rr._drone_geoms
                        and g1 in rr._table_geoms)):
                rr._table_hits += 1
                table_hit = True
            if not obj_hit and (
                    (g1 in rr._strike_geoms and g2 in rr._obj_geoms)
                    or (g2 in rr._strike_geoms
                        and g1 in rr._obj_geoms)):
                rr._obj_hits += 1
                obj_hit = True
            if ((g1 in rr._drone_geoms and g2 in rr._gate_geoms)
                    or (g2 in rr._drone_geoms
                        and g1 in rr._gate_geoms)):
                rr._gate_hits += 1
