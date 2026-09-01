"""collect_v2.py -- the v2 collection: scene, expert choreography and demo
mode, per v2_design/CONFIRMATION.md as amended by Hana (2026-09-01):

  - the goal BOX sits NEXT TO THE TABLE (replaces both the far-spawned
    bin and the earlier taped-square proposal);
  - camera3 is the measured workspace camera (pos (0, 2.25, 1.22), fovy
    52 -- v2_design/camera_pose.py, all envelope scenes PASS), rendered
    as a free camera so no XML that the frozen evaluation imports is
    edited;
  - the PENGUIN IS BLUE (dataset_tests Test B: both objects rendered
    near-black and colour never separated them; recoloured at model
    load, the v1 XML is untouched);
  - the terminal approach NEVER PAUSES: one slow forward creep with the
    grip closing IN MOTION at 15 mm -- no stillness gates, no composure
    beat, no press-retreat-repress retry (v1 measured: zero command for
    a median 107 ticks before the close, which taught the policy to
    park);
  - every leg is HOVER -> YAW IN PLACE -> TRANSLATE nose-first (Hana
    2026-09-01), so the heading channel -- the measured bottleneck,
    non-zero in only 13% of v1 frames -- is exercised on every leg,
    toward the object and again toward the box;
  - spawn envelope |x| <= 0.68 with the target sweep +-0.45 and
    COIN-FLIP target assignment (position stays uninformative by
    construction); drone start altitude widened to U(0.21, 0.60).

New file: imports collect_airvla (never edits it). Demo mode renders
sign-off episodes locally before any cluster collection:

    python collect_v2.py demo <out.mp4>
"""
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio

import collect_airvla as A

# pulled back 0.30 m (Hana, 2026-09-01 demo review) -- wider view of
# the workspace; costs some object pixels vs the 2.25 pose (measured
# 19-36 px weight at 2.50 in the sign-off sweep, still >= the 20 px bar)
CAM3_POS = np.array([0.00, 2.55, 1.30])
CAM3_LOOK = np.array([0.05, 0.08, 0.30])
CAM3_FOVY = 52.0
PENGUIN_BLUE = (0.13, 0.33, 0.82, 1.0)     # base/body/head; belly+beak keep
CLOSE_FIRE_D = 0.015                       # grip fires in motion at 15 mm
BIN_GAP = 0.45                             # box centre this far off the
                                           # table edge: ~22 cm clear gap
                                           # (Hana: laterally further out;
                                           # was flush at 0.24)


class V2Runner(A.Runner):
    def __init__(self, seed):
        super().__init__(seed)
        m = self.model
        # blue penguin (recolour at load; XML untouched)
        for g in ("penguin_base", "penguin_body", "penguin_head"):
            m.geom(g).rgba = PENGUIN_BLUE
        # legs shortened to 60% of original (Hana, two rounds: -20%
        # then "further"): hip attachment fixed, top of leg stays at
        # z=+0.020. At 0.6 the legs reach 6.2 cm below the body, less
        # than the tucked jaws -- fine airborne (episodes never land),
        # but a grounded airframe would rest on its jaws.
        for g in ("leg_front_left", "leg_front_right",
                  "leg_back_left", "leg_back_right"):
            gm = m.geom(g)
            top = float(gm.pos[2]) + float(gm.size[1])
            gm.size[1] = gm.size[1] * 0.6
            gm.pos[2] = top - float(gm.size[1])
        # geom sets for the table-clip episode gate
        tb = m.body("table").id
        self._table_geoms = set(
            g for g in range(m.ngeom)
            if m.body_rootid[m.geom_bodyid[g]] == tb)
        rb = m.body_rootid[m.geom_bodyid[m.geom("leg_front_left").id]]
        self._drone_geoms = set(
            g for g in range(m.ngeom)
            if m.body_rootid[m.geom_bodyid[g]] == rb)
        # OBJECT-STRIKE gate (Hana: the body crashed into the penguin
        # on a missed grasp): any drone-object contact NOT via the
        # gripper (clamp bodies + their pads) disqualifies the episode
        grip_bodies = {m.body(n).id for n in ("clamp_1", "clamp_2",
                                              "gripper_assembly")}
        self._strike_geoms = set(
            g for g in self._drone_geoms
            if m.geom_bodyid[g] not in grip_bodies)
        self._obj_geoms = set()
        for bn in ("pick_weight", "penguin"):
            b = m.body(bn).id
            self._obj_geoms |= set(
                g for g in range(m.ngeom)
                if m.body_rootid[m.geom_bodyid[g]] == b)
        self._obj_hits = 0
        self._table_hits = 0
        # measured workspace camera as a free camera
        self.cam3 = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam3)
        self.cam3.type = mujoco.mjtCamera.mjCAMERA_FREE
        v = CAM3_LOOK - CAM3_POS
        self.cam3.lookat[:] = CAM3_LOOK
        self.cam3.distance = float(np.linalg.norm(v))
        self.cam3.azimuth = float(np.degrees(np.arctan2(v[1], v[0])))
        self.cam3.elevation = float(np.degrees(
            np.arcsin(v[2] / self.cam3.distance)))
        self._fovy0 = float(m.vis.global_.fovy)
        self.bin_id = m.body("bin").id

    def step(self, frames, task):
        super().step(frames, task)
        # episode gate (Hana): any drone-table contact disqualifies the
        # episode -- counted per tick, checked at banking time
        d = self.data
        table_hit = obj_hit = False
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if not table_hit and (
                    (g1 in self._drone_geoms and g2 in self._table_geoms)
                    or (g2 in self._drone_geoms
                        and g1 in self._table_geoms)):
                self._table_hits += 1
                table_hit = True
            if not obj_hit and (
                    (g1 in self._strike_geoms and g2 in self._obj_geoms)
                    or (g2 in self._strike_geoms
                        and g1 in self._obj_geoms)):
                self._obj_hits += 1
                obj_hit = True
            if table_hit and obj_hit:
                break

    def frame(self, task):
        f = {"observation.state": self.state(),
             "scene_state": self.scene_state(),
             "action": None, "task": task}
        for key, cam in A.CAMS.items():
            if key == "camera3":
                self.model.vis.global_.fovy = CAM3_FOVY
                self.rend.update_scene(self.data, camera=self.cam3)
                self.model.vis.global_.fovy = self._fovy0
            else:
                self.rend.update_scene(self.data, camera=cam)
            f[f"observation.images.{key}"] = self.rend.render().copy()
        return f

    # -- v2 scene ----------------------------------------------------------
    def reset_scene_v2(self, obj=None, corrective=False):
        rng = self.rng
        # pair inside the frame-safe envelope; assignment by coin flip
        dsep = rng.uniform(0.45, 0.62)
        cx = rng.uniform(-(0.68 - dsep / 2), 0.68 - dsep / 2)
        cy = rng.uniform(0.20, 1.00)
        spots = [np.array([cx - dsep / 2, cy]), np.array([cx + dsep / 2, cy])]
        rng.shuffle(spots)
        if obj is None:
            obj = "weight" if rng.random() < 0.5 else "plush penguin"
        tgt_xy, dis_xy = spots
        alt = (A.PLATE_TOP + 0.005 + self.objs[obj]["aim_z"]
               - float(self.expert.off_carry[2]))
        for _ in range(300):
            start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.6),
                              rng.uniform(0.21, 0.60)])
            if (np.linalg.norm(start[0:2] - tgt_xy) >= 0.70
                    and start[1] >= tgt_xy[1] + 0.35):
                break
        self.reset_scene(tgt_xy, start, obj=obj)
        # distractor to ITS coin-flipped spot (overrides the side rule)
        other = [o for k, o in self.objs.items() if k != obj][0]
        self.data.qpos[other["adr"]:other["adr"] + 2] = dis_xy
        # the BOX sits NEXT TO THE TABLE (Hana): off the table edge on
        # the side away from the pair centre, clear of the legs
        m = self.model
        side = -1.0 if cx >= self._table_cx else 1.0
        bxy = np.array([self._table_cx + side * (0.45 + BIN_GAP),
                        float(m.body_pos[self.model.body("table").id][1])])
        self.bin_xy = bxy
        m.body_pos[self.bin_id][0:2] = bxy
        m.body_pos[self.bin_id][2] = A.MAT_TOP
        # ALIGNED with the table (Hana): the v1 scene rotates the bin
        # randomly; squared-with-the-table reads as aligned
        m.body_quat[self.bin_id] = [1, 0, 0, 0]
        self.bin_yaw = 0.0
        self._table_hits = 0
        self._obj_hits = 0
        mujoco.mj_forward(m, self.data)
        return obj, start, alt, tgt_xy

    def hold_xy_v2(self, frames, task, base_xy, z_target, done,
                   timeout_s):
        """Trim-compensated station-keeping with a LOW-PASS on the
        correction (one pole, 0.25/tick): the raw per-tick corr
        feedback chatters at 10 Hz -- visible as hover wobble (Hana),
        worst right after the weld transient."""
        ex = self.expert
        corr_f = self.data.qpos[0:2] - ex.sp[0:2]
        for _ in range(int(timeout_s * A.FPS)):
            corr = self.data.qpos[0:2] - ex.sp[0:2]
            corr_f = 0.75 * corr_f + 0.25 * corr
            ex.set_goal(xyz=np.array([base_xy[0] - corr_f[0],
                                      base_xy[1] - corr_f[1],
                                      float(z_target)]))
            self.step(frames, task)
            if done():
                return True
        return False

    # -- v2 choreography ---------------------------------------------------
    def yaw_then_go(self, frames, task, to_xy, min_turn=0.03):
        """Hover -> yaw in place toward to_xy -> return the unit heading.
        The translation goal pins the CURRENT setpoint, so the drone
        turns on its own axis before any leg begins."""
        ex = self.expert
        here = self.data.qpos[0:2]
        v = np.array(to_xy) - here
        if float(np.linalg.norm(v)) > 0.05:
            ex.set_goal(yaw=float(np.arctan2(v[0], -v[1])))
            self.run_until(frames, task,
                           lambda: abs(ex.yaw - ex.goal_yaw) < min_turn,
                           timeout_s=8.0)
        return v / max(1e-9, float(np.linalg.norm(v)))

    def pick_v2(self, frames, task, alt, corrective=False):
        """Continuous-terminal pick: no stillness gates, no composure
        beat, no retry loop. Grip fires IN MOTION at CLOSE_FIRE_D."""
        ex = self.expert
        aim, _ = self.live_target()
        if corrective:
            # FLOWN displaced approach: a teleported hover proved
            # pathological twice (stale controller state, arm-swing
            # lurch; runaways of 1.2 m and 2.2 m). The drone flies the
            # normal ascent and turn, but its pre-grasp point is
            # DISPLACED 5-20 cm at a random room-side bearing; the
            # creep then demonstrates the correction the v1 expert
            # never showed.
            ex.set_goal(xyz=np.array([ex.sp[0], ex.sp[1], alt]))
            self.settle_near(frames, task, tol=0.05, timeout_s=8.0)
            th = self.rng.uniform(-1.25, 1.25)
            r_ = self.rng.uniform(0.05, 0.20)
            wrong = aim[0:2] + r_ * np.array([np.sin(th), -np.cos(th)])
            here = self.data.qpos[0:2]
            u0 = wrong - here
            u0 = u0 / max(1e-9, float(np.linalg.norm(u0)))
            u0 = u0 + np.array([0.0, -1.4])
            u0 = u0 / float(np.linalg.norm(u0))
            # displaced leg flies 6 cm high: the corrective creep can
            # cross the tabletop diagonally, and at grasp height the
            # jaw pads graze it (measured: 28 pad-contact ticks); the
            # creep's P-arrival then descends into the grasp
            pre = np.array([*(wrong - 0.20 * u0), alt + 0.06])
            self.yaw_then_go(frames, task, pre[0:2])
            ex.set_goal(xyz=pre, arm=ex.q_travel)
            self.settle_near(frames, task, tol=0.05, timeout_s=14.0)
            self.yaw_then_go(frames, task, aim[0:2])
            ex.set_goal(arm=ex.q_carry)
            self.run_until(frames, task,
                           lambda: float(np.linalg.norm(
                               self.data.qpos[7:9] - ex.q_carry)) < 0.06,
                           timeout_s=10.0)
        else:
            # ascend at the spawn, then yaw-in-place, then one straight
            # nose-first leg to the pre-grasp point at grasp altitude
            ex.set_goal(xyz=np.array([ex.sp[0], ex.sp[1], alt]))
            self.settle_near(frames, task, tol=0.05, timeout_s=8.0)
            # arm deploys AT THE SPAWN HOVER, far from the table: the
            # deployment swing (CoM shift) was the wobble at the
            # table-edge hover; it now happens over open floor
            ex.set_goal(arm=ex.q_carry)
            self.run_until(frames, task,
                           lambda: float(np.linalg.norm(
                               self.data.qpos[7:9] - ex.q_carry)) < 0.06,
                           timeout_s=10.0)
            held0 = [0]

            def spawn_calm():
                ok0 = (float(np.linalg.norm(self.data.qvel[0:3])) < 0.05
                       and float(np.linalg.norm(
                           self.data.qvel[3:6])) < 0.12)
                held0[0] = held0[0] + 1 if ok0 else 0
                return held0[0] >= 5
            self.hold_xy_v2(frames, task, self.data.qpos[0:2].copy(),
                            alt, spawn_calm, timeout_s=6.0)
            # approach direction biased toward the front-edge normal
            # (from the room side, +y): a diagonal arrival stands the
            # body over the tabletop and the legs catch the table edge
            # -- measured as a 93 mm stall on the low-stemmed weight
            here = self.data.qpos[0:2]
            u0 = aim[0:2] - here
            u0 = u0 / max(1e-9, float(np.linalg.norm(u0)))
            u0 = u0 + np.array([0.0, -1.4])
            u0 = u0 / float(np.linalg.norm(u0))
            # standoff widened 0.35 -> 0.50 m (Hana: hover further
            # from the table edge); the tapered creep below keeps the
            # longer approach quick and pause-free
            pre = aim.copy()
            pre[0:2] -= 0.50 * u0
            pre[2] = alt
            self.yaw_then_go(frames, task, pre[0:2])
            # arm already deployed at the spawn; the standoff hover is
            # a calm gate, not a reconfiguration point
            ex.set_goal(xyz=pre)
            self.settle_near(frames, task, tol=0.05, timeout_s=14.0)
            self.yaw_then_go(frames, task, aim[0:2])
            held0 = [0]

            def pre_calm():
                ok0 = (float(np.linalg.norm(self.data.qvel[0:3])) < 0.05
                       and float(np.linalg.norm(
                           self.data.qvel[3:6])) < 0.12)
                held0[0] = held0[0] + 1 if ok0 else 0
                return held0[0] >= 5
            self.hold_xy_v2(frames, task, self.data.qpos[0:2].copy(),
                            alt, pre_calm, timeout_s=6.0)
        # continuous creep at 3 mm/tick: goal slightly PAST dead-centre
        # so the setpoint is still moving when the grip fires. The fire
        # distance covers the travel DURING the close ramp (the first
        # demo fired at a flat 15 mm: the wide penguin head forgave the
        # ramp-time overshoot, the 16 mm weight stem did not), and the
        # weld gate is checked EVERY tick of the ramp so the grasp
        # seats at the first centred, in-aperture-window instant.
        close = self.cur["close"]
        ramp_ticks = (1.0 - close) / A.GRIP_STEP
        creep = 0.003
        fire_d = 0.002 + creep * ramp_ticks
        fired = [False]
        welded = [False]

        u_fix = [None]                   # approach axis frozen at start
        fire_tick = [None]
        tick_n = [0]

        def creep_tick():
            tick_n[0] += 1
            aim_l, _ = self.live_target()
            cyw, syw = np.cos(ex.yaw), np.sin(ex.yaw)
            offw = np.array([cyw * ex.off_carry[0] - syw * ex.off_carry[1],
                             syw * ex.off_carry[0] + cyw * ex.off_carry[1],
                             ex.off_carry[2]])
            u = aim_l[0:2] - self.jaws()[0:2]
            d = float(np.linalg.norm(u))
            diag["dmin"] = min(diag["dmin"], d)
            diag["fired"] = fired[0]
            # distance-tapered creep: 8 mm/tick far, at the 3 mm/tick
            # floor BEFORE the table edge (~0.15 m out) -- at 10 mm/tick
            # the nose-down pitch of faster flight reached the front
            # leg into the table edge (measured: contacts from tick
            # ~210 in both standard episodes; 3 mm/tick round 3 was
            # clean over the identical final geometry)
            A.SP_STEP_FINAL = float(np.clip(0.12 * (d - 0.13) + creep,
                                            creep, 0.008))
            if u_fix[0] is None:
                u_fix[0] = u / max(1e-9, d)
            goal = aim_l - offw
            # overshoot along the FROZEN approach axis: the live axis
            # flips sign once the jaws pass the object, and the
            # re-aimed overshoot ratcheted the body onto the penguin
            # on a missed close (Hana's crash report)
            goal[0:2] += 0.012 * u_fix[0]
            ex.set_goal(xyz=goal)
            if not fired[0] and d < fire_d:
                fired[0] = True
                fire_tick[0] = tick_n[0]
                ex.set_goal(grip=close)
            if (fired[0] and not welded[0]
                    and tick_n[0] - fire_tick[0] > 25):
                raise StopIteration     # missed close: abort, discard
            if fired[0] and not welded[0]:
                pc = self.jaws() - aim_l
                ap = float(self.data.qpos[self.gadr])
                if (float(np.linalg.norm(pc[0:2])) < 0.010
                        and abs(float(pc[2])) < 0.015
                        and float(np.linalg.norm(
                            self.data.qvel[0:3])) < 0.06
                        and self.cur["ap_lo"] < ap * 1000 < self.cur["ap_hi"]):
                    welded[0] = True
                    self.weld_grasp(True)
            return welded[0]

        diag = dict(dmin=9e9)
        self._creep_diag = diag
        ex.slow = True
        sp_final0 = A.SP_STEP_FINAL
        A.SP_STEP_FINAL = creep          # terminal law floor, this leg only
        try:
            ok = self.run_until(frames, task, creep_tick, timeout_s=25.0,
                                min_hold=1)
        except StopIteration:
            ok = False                  # banking filter discards this one
        finally:
            A.SP_STEP_FINAL = sp_final0
            ex.slow = False
        if ok:
            # POST-GRASP STABILIZATION (Hana): the weld transient plus
            # residual creep momentum excites a swing. Freeze the goal
            # at the current setpoint (kill the +12 mm overshoot), then
            # hold station trim-compensated until BOTH linear and
            # angular rates are low for 5 consecutive ticks. This is
            # after the close, so the parking-window metric (which
            # measures the pre-close command) is untouched.
            ex.set_goal(xyz=ex.sp.copy())
            held = [0]

            def calm():
                still = (float(np.linalg.norm(self.data.qvel[0:3])) < 0.05
                         and float(np.linalg.norm(
                             self.data.qvel[3:6])) < 0.12)
                held[0] = held[0] + 1 if still else 0
                return held[0] >= 5
            self.hold_xy_v2(frames, task, self.data.qpos[0:2].copy(),
                               float(ex.sp[2]), calm, timeout_s=5.0)
        return ok

    def place_v2(self, frames, task):
        """Lift -> yaw in place toward the box -> trim-compensated carry
        -> lower over the box -> release. The loaded PD parks ~0.28 m
        past a plain setpoint (v1 lesson: lead compensation is
        load-bearing), so the carry and descent run through
        hold_xy_until, which aims the setpoint short by the live trim
        estimate -- the first demo used settle_near and the drone
        wandered off-mat, releasing 6 m from the box."""
        ex = self.expert
        # EVERY loaded phase is trim-compensated (hold_xy_until): the
        # payload's CoM shift turns a plain fixed setpoint into a
        # constant-acceleration slide -- traced numerically: the body
        # bolted 3 m during a settle_near lift before the carry began
        here0 = self.data.qpos[0:2].copy()
        lift_z = float(self.data.qpos[2]) + 0.30
        held = [0]

        def lifted_calm():
            ok_ = (abs(float(self.data.qpos[2]) - lift_z) < 0.06
                   and float(np.linalg.norm(self.data.qvel[0:3])) < 0.05
                   and float(np.linalg.norm(self.data.qvel[3:6])) < 0.12)
            held[0] = held[0] + 1 if ok_ else 0
            return held[0] >= 5
        self.hold_xy_v2(frames, task, here0, lift_z, lifted_calm,
                           timeout_s=12.0)
        v = self.bin_xy - self.data.qpos[0:2]
        ex.set_goal(yaw=float(np.arctan2(v[0], -v[1])))
        self.hold_xy_v2(
            frames, task, here0, lift_z,
            lambda: abs(ex.yaw - ex.goal_yaw) < 0.03,
            timeout_s=8.0)
        # the OBJECT hangs at the carry offset from the body: station
        # the BODY so the JAWS sit over the box (the 193 mm systematic
        # miss of demo 2 was exactly this offset)
        cyw, syw = np.cos(ex.yaw), np.sin(ex.yaw)
        offw = np.array([cyw * ex.off_carry[0] - syw * ex.off_carry[1],
                         syw * ex.off_carry[0] + cyw * ex.off_carry[1]])
        base = self.bin_xy - offw
        # constant-altitude carry (Hana: descending while translating
        # makes the loaded drone drift): fly AT lift_z to above the
        # box, only then descend vertically
        self.hold_xy_v2(
            frames, task, base, lift_z,
            lambda: float(np.linalg.norm(
                self.jaws()[0:2] - self.bin_xy)) < 0.05,
            timeout_s=20.0)
        # release height keeps the LEGS above the tabletop: the box is
        # flush with the table, so a descent below tabletop level parks
        # the table-side legs into the edge (measured: 6-13 leg-contact
        # ticks in the final descent of every first-gate demo episode)
        low_z = A.PLATE_TOP + 0.12
        self.hold_xy_v2(
            frames, task, base, low_z,
            lambda: abs(float(self.data.qpos[2]) - low_z) < 0.05
            and float(np.linalg.norm(
                self.jaws()[0:2] - self.bin_xy)) < 0.05,
            timeout_s=10.0)
        self.weld_grasp(False)
        ex.set_goal(grip=1.0)
        self.run_until(frames, task, lambda: ex.grip > 0.99, timeout_s=2.0)
        # POST-RELEASE RECOVERY (Hana: the drone clipped the table and
        # tumbled after the drop): losing the payload INVERTS the trim
        # the controller had learned to lean against, and the lurch can
        # drive the body into the table edge below. Climb immediately
        # to a safe altitude over the box and station-hold until both
        # rates are calm for 5 consecutive ticks -- inside the recorded,
        # table-clip-gated window.
        safe_z = low_z + 0.30
        here = self.data.qpos[0:2].copy()
        held = [0]

        def recovered():
            ok_ = (abs(float(self.data.qpos[2]) - safe_z) < 0.08
                   and float(np.linalg.norm(self.data.qvel[0:3])) < 0.06
                   and float(np.linalg.norm(self.data.qvel[3:6])) < 0.15)
            held[0] = held[0] + 1 if ok_ else 0
            return held[0] >= 5
        self.hold_xy_v2(frames, task, here, safe_z, recovered,
                           timeout_s=8.0)
        adr = self.cur["adr"]
        oxy = self.data.qpos[adr:adr + 2]
        return float(np.linalg.norm(oxy - self.bin_xy))

    def episode_v2(self, corrective=False, obj=None):
        obj, start, alt, tgt = self.reset_scene_v2(obj=obj,
                                                   corrective=corrective)
        task = A.PROMPT_MANIP.format(obj=obj)
        frames = []
        grasped = self.pick_v2(frames, task, alt, corrective=corrective)
        d_bin = self.place_v2(frames, task) if grasped else float("nan")
        return frames, dict(obj=obj, corrective=corrective,
                            grasped=grasped, d_bin_mm=round(1000 * d_bin, 1)
                            if grasped else None, ticks=len(frames),
                            table_hits=self._table_hits,
                            obj_hits=self._obj_hits,
                            clean=(self._table_hits == 0
                                   and self._obj_hits == 0))


def parking_window(frames):
    """Self-check: ticks between the last non-zero horizontal command
    and the grip-close crossing. The v2 acceptance number is ~0."""
    act = np.stack([f["action"] for f in frames])
    grip = act[:, 6]
    cr = np.where((grip[1:] < 0.9) & (grip[:-1] >= 0.9))[0] + 1
    if len(cr) == 0:
        return None
    tg = int(cr[-1])
    nz = np.where(np.linalg.norm(act[:tg, 0:2], axis=1) > 1e-9)[0]
    return tg - int(nz[-1]) if len(nz) else tg


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    assert mode == "demo"
    r = V2Runner(31000)
    w = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                           macro_block_size=1)
    for corrective, obj in ((False, "weight"), (False, "plush penguin"),
                            (True, "plush penguin")):
        frames, res = r.episode_v2(corrective=corrective, obj=obj)
        pw = parking_window(frames)
        print("EP %-14s corrective=%-5s grasped=%-5s d_bin=%s mm  "
              "parking window=%s ticks  table_hits=%d obj_hits=%d %s "
              "(%d ticks; creep dmin %.1f mm)"
              % (res["obj"], corrective, res["grasped"], res["d_bin_mm"],
                 pw, res["table_hits"], res["obj_hits"],
                 "CLEAN" if res["clean"] else "REJECT",
                 res["ticks"], 1000 * r._creep_diag["dmin"]), flush=True)
        for f in frames[::2]:
            img = np.concatenate(
                [f["observation.images.camera1"],
                 f["observation.images.camera2"],
                 f["observation.images.camera3"]], axis=1)
            w.append_data(img)
    w.close()
    print("demo ->", out)
