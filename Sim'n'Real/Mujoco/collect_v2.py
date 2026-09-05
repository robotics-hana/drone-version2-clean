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
        # legs at 35% of original (Hana, three rounds of shortening;
        # the third enables 2D on-table object spawns): leg tips now
        # sit ~1 cm ABOVE tabletop height at grasp altitude, so the
        # body can cross the table without the legs reaching the top.
        # Airborne-only consequence unchanged: grounded, it rests on
        # its jaws.
        for g in ("leg_front_left", "leg_front_right",
                  "leg_back_left", "leg_back_right"):
            gm = m.geom(g)
            top = float(gm.pos[2]) + float(gm.size[1])
            gm.size[1] = gm.size[1] * 0.35
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
        # OBJECT-STRIKE gate: any drone-object contact NOT via the
        # gripper (clamp bodies) disqualifies the episode
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
        self._grip_geoms = set(
            g for g in self._drone_geoms
            if m.geom_bodyid[g] in grip_bodies)
        self._grip_contact_ok = False    # True only creep -> release
        gt = m.body("gate").id
        self._gate_geoms = set(
            g for g in range(m.ngeom)
            if m.body_rootid[m.geom_bodyid[g]] == gt)
        self._gate_hits = 0
        self._obj_hits = 0
        self._table_hits = 0
        self._ff_on = False
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

    def weld_grasp(self, on):
        """Payload feed-forward (Hana: 'what causes the thrust motion
        when an item is picked up?'): the controller's hover thrust is
        computed for the UNLOADED mass (pd_flight.py: total_mass*9.81),
        so at the weld the drone sags and pitches until the error-driven
        PD catches up -- the visible surge. Telling the controller about
        the payload at the weld instant (and taking it back at release)
        means the sag never develops."""
        m = self.model
        if on and not self._ff_on:
            body = self.cur["body"]
            sub = float(m.body_subtreemass[body])
            self.ctrl.mppi.nominal_hover_thrust += sub * 9.81
            self._ff_on = True
        elif not on and self._ff_on:
            body = self.cur["body"]
            sub = float(m.body_subtreemass[body])
            self.ctrl.mppi.nominal_hover_thrust -= sub * 9.81
            self._ff_on = False
        super().weld_grasp(on)

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
            if not obj_hit and not self._grip_contact_ok and (
                    (g1 in self._grip_geoms and g2 in self._obj_geoms)
                    or (g2 in self._grip_geoms
                        and g1 in self._obj_geoms)):
                # jaws touching an object OUTSIDE the grasp phase is a
                # strike too (the corrective's turn swept the penguin)
                self._obj_hits += 1
                obj_hit = True
            if ((g1 in self._drone_geoms and g2 in self._gate_geoms)
                    or (g2 in self._drone_geoms
                        and g1 in self._gate_geoms)):
                self._gate_hits += 1
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
    TABLE_C = np.array([0.0, 0.50])    # FIXED table centre (Hana:
                                       # constant framing in camera3)
    def reset_scene_v2(self, obj=None, corrective=False, nav=False,
                       layout=None):
        rng = self.rng
        fe = float(self.TABLE_C[1]) + 0.30          # front edge y
        if layout is not None:
            # PAIRED-COMMAND replay (E3 grounding component, Hana
            # 2026-09-05): re-create a previous episode's scene
            # EXACTLY -- object positions, spawn, box side -- with
            # only the commanded object changed, so the instruction
            # is the sole signal distinguishing the two trajectories.
            assert obj is not None and not nav and not corrective
            other_n = [k for k in self.objs if k != obj][0]
            tgt_xy = np.array(layout["pos"][obj], dtype=float)
            dis_xy = np.array(layout["pos"][other_n], dtype=float)
            start = np.array(layout["start"], dtype=float)
            side = float(layout["side"])
        else:
            # both objects ON the fixed table, positions varied in 2D
            # (x +-0.35 of centre, 6-26 cm inside the front edge),
            # >=0.40 m apart, target assigned by COIN FLIP -- position
            # stays uninformative by construction.
            # nav episodes have no grasp-reach constraint, so their
            # objects roam the FULL usable table depth (Hana: vary
            # positions so the model cannot overfit a spot); picks
            # keep the reachable band
            y_lo = fe - 0.52 if nav else fe - 0.26
            for _ in range(200):
                spots = [np.array([rng.uniform(-0.35, 0.35)
                                   + self.TABLE_C[0],
                                   rng.uniform(y_lo, fe - 0.06)])
                         for _ in range(2)]
                if float(np.linalg.norm(spots[0] - spots[1])) >= 0.40:
                    break
            rng.shuffle(spots)
            if obj is None:
                obj = "weight" if rng.random() < 0.5 else "plush penguin"
            tgt_xy, dis_xy = spots
        alt = (A.PLATE_TOP + 0.005 + self.objs[obj]["aim_z"]
               - float(self.expert.off_carry[2]))
        if nav:
            # nav: spawn SOUTH of the gate facing the room (v1 eval
            # geometry), gate side coin-flipped
            gx = -0.7 if rng.random() < 0.5 else 0.7
            start = np.array([rng.uniform(-0.95, 0.95),
                              rng.uniform(-1.9, -1.2),
                              rng.uniform(0.60, 1.00)])
            yaw0 = np.pi + rng.uniform(-0.26, 0.26)
            self.reset_scene(tgt_xy, start,
                             gate_xy=np.array([gx, -0.6]),
                             yaw=yaw0, obj=obj)
            self._gate_x = gx
        else:
            if layout is None:
                for _ in range(300):
                    start = np.array([rng.uniform(-0.5, 0.5),
                                      rng.uniform(1.1, 1.7),
                                      rng.uniform(0.21, 0.60)])
                    if np.linalg.norm(start[0:2] - tgt_xy) >= 0.70:
                        break
            self.reset_scene(tgt_xy, start, obj=obj)
        m = self.model
        # FIXED table (reset_scene slid it under the task spot; put it
        # back): constant framing in camera3, every episode
        m.body_pos[m.body("table").id][0:2] = self.TABLE_C
        self._table_cx = float(self.TABLE_C[0])
        # both objects to their sampled on-table spots
        curo = self.objs[obj]
        self.data.qpos[curo["adr"]:curo["adr"] + 2] = tgt_xy
        other = [o for k, o in self.objs.items() if k != obj][0]
        self.data.qpos[other["adr"]:other["adr"] + 2] = dis_xy
        # the BOX beside the fixed table, side coin-flipped, both sides
        # inside the measured camera3 frame (pinned when replaying a
        # paired layout)
        if layout is None:
            side = -1.0 if rng.random() < 0.5 else 1.0
        bxy = np.array([self.TABLE_C[0] + side * (0.45 + BIN_GAP),
                        float(self.TABLE_C[1])])
        self.bin_xy = bxy
        m.body_pos[self.bin_id][0:2] = bxy
        m.body_pos[self.bin_id][2] = A.MAT_TOP
        # ALIGNED with the table (Hana): the v1 scene rotates the bin
        # randomly; squared-with-the-table reads as aligned
        m.body_quat[self.bin_id] = [1, 0, 0, 0]
        self.bin_yaw = 0.0
        self._table_hits = 0
        self._obj_hits = 0
        self._gate_hits = 0
        if not nav:
            # record the layout so a paired episode can replay it
            # with the other command (E3 grounding component)
            other_n = [k for k in self.objs if k != obj][0]
            self.last_layout = dict(
                pos={obj: [float(tgt_xy[0]), float(tgt_xy[1])],
                     other_n: [float(dis_xy[0]), float(dis_xy[1])]},
                start=[float(x) for x in start], side=float(side))
        mujoco.mj_forward(m, self.data)
        return obj, start, alt, tgt_xy

    def hold_xy_v2(self, frames, task, base_xy, z_target, done,
                   timeout_s):
        """Trim-compensated station-keeping with a LOW-PASS on the
        correction (one pole, 0.25/tick, seeded from the live value):
        the raw per-tick corr feedback chatters at 10 Hz."""
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

    ARM_CLEAR = 0.35   # arm-extension clearance from the table rect:
                       # jaw forward reach (~0.20) + margin (Hana: the
                       # drone must be at least an arm away from the
                       # table whenever the arm extends -- ep2/ep4 of
                       # the 10-episode demo struck the table there)

    def table_clearance(self, p_xy):
        lo = self.TABLE_C - np.array([0.45, 0.30])
        hi = self.TABLE_C + np.array([0.45, 0.30])
        d = np.maximum(np.maximum(lo - p_xy, p_xy - hi), 0.0)
        return float(np.linalg.norm(d))

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

    def pick_v2(self, frames, task, alt, corrective=False,
                terminal=False):
        """Continuous-terminal pick: no stillness gates, no composure
        beat, no retry loop. Grip fires IN MOTION at CLOSE_FIRE_D.

        terminal=True (E3 flavour, pre-registered 2026-09-05): after
        the arm deploys, fly to a PERTURBED near-miss hover first --
        the policy's observed failure distribution (2-4 cm lateral,
        +3..+13 cm high) -- hesitate 10-20 ticks, then let the normal
        creep correct it. Supervises 'persist near the object and
        re-align', which no other flavour contains. Default False =
        every existing flavour's path is untouched."""
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
            # turn-in-place happens 0.40 m out (was 0.20): the jaw
            # lever reaches ~0.2 m, so a turn or arm extension at the
            # old standoff swept the jaws into the object (Hana saw the
            # corrective hit the penguin -- a GRIPPER contact, which
            # the strike gate deliberately allowed; see phase rule)
            # LEVEL at grasp altitude (the +0.06 was cover for the old
            # 0.20 m standoff's pad-drag; at 0.40 m the geometry equals
            # the standard branch, and the high start left the jaws
            # 14 mm above the weld gate at the window transit)
            # +0.03: level flight grazed the table edge with the
            # TUCKED jaws (one tick); half the old offset clears them
            # while the residual height at fire (~7 mm) stays inside
            # the 15 mm weld gate
            pre = np.array([*(wrong - 0.40 * u0), alt + 0.03])
            back = 0.40
            while (self.table_clearance(pre[0:2]) < self.ARM_CLEAR
                   and back < 1.0):
                back += 0.05
                pre[0:2] = wrong - back * u0
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
            # approach direction biased toward the front-edge normal
            # (from the room side, +y): a diagonal arrival stands the
            # body over the tabletop and the legs catch the table edge
            # -- measured as a 93 mm stall on the low-stemmed weight
            here = self.data.qpos[0:2]
            u0 = aim[0:2] - here
            u0 = u0 / max(1e-9, float(np.linalg.norm(u0)))
            u0 = u0 + np.array([0.0, -1.4])
            u0 = u0 / float(np.linalg.norm(u0))
            # standoff 0.45 (was 0.35): the turn toward the object
            # must finish outside the jaw-sweep radius plus margin
            pre = aim.copy()
            pre[0:2] -= 0.45 * u0
            pre[2] = alt
            # HARD RULE: never extend the arm closer than ARM_CLEAR to
            # the table -- push the standoff outward until it clears
            back = 0.45
            while (self.table_clearance(pre[0:2]) < self.ARM_CLEAR
                   and back < 1.0):
                back += 0.05
                pre[0:2] = aim[0:2] - back * u0
            self.yaw_then_go(frames, task, pre[0:2])
            ex.set_goal(xyz=pre, arm=ex.q_travel)
            self.settle_near(frames, task, tol=0.05, timeout_s=14.0)
            self.yaw_then_go(frames, task, aim[0:2])
            ex.set_goal(arm=ex.q_carry)
            self.run_until(frames, task,
                           lambda: float(np.linalg.norm(
                               self.data.qpos[7:9] - ex.q_carry)) < 0.06,
                           timeout_s=10.0)
        # E3 terminal-corrective stage (design: Hana, 2026-09-05
        # review). The episode drifts off the object line the way the
        # POLICY does -- a level nose-first leg toward a FALSE point
        # laterally offset by the MEASURED failure distribution
        # (near-cluster lateral median 32 mm, range ~20-55 mm) and
        # 6-10 cm short of the object -- then corrects with the
        # standard v2 vocabulary: yaw-in-place to re-point the
        # gripper at the object, then the untouched normal creep,
        # fire and grasp. No settle wait, no dwell, no vertical
        # plunge, no non-standard motion primitive; the stock creep
        # below is byte-identical for every flavour.
        if terminal:
            aim_t, _ = self.live_target()
            u_app = aim_t[0:2] - self.jaws()[0:2]
            u_app = u_app / max(1e-9, float(np.linalg.norm(u_app)))
            perp = np.array([-u_app[1], u_app[0]])
            side = 1.0 if float(self.rng.uniform()) < 0.5 else -1.0
            lat = float(self.rng.uniform(0.02, 0.055))
            # 10-16 cm short (was 6-10): the post-yaw creep needs
            # runway for the lateral offset to converge before the
            # jaws pass the object -- at 6-10 cm the pass-through
            # guard correctly aborted (~65-79 mm stalls, diag
            # 2026-09-06); the standard creep gets 45 cm
            short = float(self.rng.uniform(0.10, 0.16))
            false_xy = (aim_t[0:2] + side * lat * perp
                        - short * u_app)
            # jaws to the false point, level at the current altitude
            cyw, syw = np.cos(ex.yaw), np.sin(ex.yaw)
            offw = np.array([cyw * ex.off_carry[0] - syw * ex.off_carry[1],
                             syw * ex.off_carry[0] + cyw * ex.off_carry[1],
                             ex.off_carry[2]])
            false_goal = np.array([*false_xy, float(ex.sp[2])]) \
                - np.array([offw[0], offw[1], 0.0])
            ex.set_goal(xyz=false_goal)
            self.run_until(frames, task,
                           lambda: float(np.linalg.norm(
                               self.jaws()[0:2] - false_xy)) < 0.03,
                           timeout_s=10.0)
            # the corrective re-yaw: point the gripper back at the
            # object (standard yaw-in-place primitive), then fall
            # through to the normal creep -> approach and grasp as
            # in every standard episode
            self.yaw_then_go(frames, task, aim_t[0:2])
        # continuous creep at 3 mm/tick: goal slightly PAST dead-centre
        # so the setpoint is still moving when the grip fires. The fire
        # distance covers the travel DURING the close ramp (the first
        # demo fired at a flat 15 mm: the wide penguin head forgave the
        # ramp-time overshoot, the 16 mm weight stem did not), and the
        # weld gate is checked EVERY tick of the ramp so the grasp
        # seats at the first centred, in-aperture-window instant.
        close = self.cur["close"]
        creep = 0.003
        # fire distance from the MEASURED finger rate (~0.8 mm/tick;
        # the physical aperture lags the 2.4 mm/tick command -- gate
        # log, diag3): the grip starts early enough that the aperture
        # window crosses exactly at jaw-centre
        win_mid = (self.cur["ap_lo"] + self.cur["ap_hi"]) / 2.0
        ramp_ticks = max(2.0, (16.0 - win_mid) / 0.8)
        fire_d = 0.004 + creep * ramp_ticks
        fired = [False]
        welded = [False]
        u_fix = [None]
        fwd_max = [None]
        z_corr = [0.0]                  # filtered jaw-sag compensation
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
            if u_fix[0] is None:
                u_fix[0] = u / max(1e-9, d)
            # GRASP ON APPROACH ONLY: jaws past the object without a
            # weld -> abort (episode discarded); never back up
            if (not welded[0] and float(
                    (self.jaws()[0:2] - aim_l[0:2]) @ u_fix[0]) > 0.018):
                raise StopIteration
            goal = aim_l - offw
            # jaw-height servo: the arm SAGS below its FK offset under
            # gravity (measured riding the 15 mm weld gate at 14.3 mm),
            # so drive the MEASURED jaw height onto the target
            z_corr[0] = (0.85 * z_corr[0]
                         + 0.15 * float((self.jaws() - aim_l)[2]))
            # RAISE-only: compensate sag (jaws below aim), never push
            # the body lower -- the filter's lag during the corrective's
            # high-start descent over-lowered the body and the legs
            # grazed the table edge (164 contacts)
            goal[2] -= min(z_corr[0], 0.0)
            goal[0:2] += 0.012 * u_fix[0]   # overshoot: frozen axis
            ga = float(goal[0:2] @ u_fix[0])
            if fwd_max[0] is not None and ga < fwd_max[0]:
                goal[0:2] += (fwd_max[0] - ga) * u_fix[0]
            fwd_max[0] = ga if fwd_max[0] is None else max(fwd_max[0], ga)
            ex.set_goal(xyz=goal)
            if not fired[0] and d < fire_d:
                fired[0] = True
                fire_tick[0] = tick_n[0]
                ex.set_goal(grip=close)
            if (fired[0] and not welded[0] and fire_tick[0] is not None
                    and tick_n[0] - fire_tick[0] > 30):
                raise StopIteration
            if fired[0] and not welded[0]:
                pc = self.jaws() - aim_l
                ap = float(self.data.qpos[self.gadr])
                if (float(np.linalg.norm(pc[0:2])) < 0.010
                        and abs(float(pc[2])) < 0.015
                        and self.cur["ap_lo"] < ap * 1000 < self.cur["ap_hi"]):
                    welded[0] = True
                    self.weld_grasp(True)
            return welded[0]

        diag = dict(dmin=9e9)
        self._creep_diag = diag
        self._grip_contact_ok = True     # grasp phase begins
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
                return held[0] >= 3
            self.hold_xy_until(frames, task, self.data.qpos[0:2].copy(),
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
            return held[0] >= 3
        self.hold_xy_until(frames, task, here0, lift_z, lifted_calm,
                           timeout_s=12.0)
        v = self.bin_xy - self.data.qpos[0:2]
        ex.set_goal(yaw=float(np.arctan2(v[0], -v[1])))
        self.hold_xy_until(
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
        drop_z = A.MAT_TOP + 0.55
        self.hold_xy_until(
            frames, task, base, drop_z,
            lambda: float(np.linalg.norm(
                self.jaws()[0:2] - self.bin_xy)) < 0.05
            and abs(float(self.data.qpos[2]) - drop_z) < 0.08,
            timeout_s=20.0)
        # release height keeps the LEGS above the tabletop: the box is
        # flush with the table, so a descent below tabletop level parks
        # the table-side legs into the edge (measured: 6-13 leg-contact
        # ticks in the final descent of every first-gate demo episode)
        low_z = A.PLATE_TOP + 0.12
        self.hold_xy_until(
            frames, task, base, low_z,
            lambda: abs(float(self.data.qpos[2]) - low_z) < 0.05
            and float(np.linalg.norm(
                self.jaws()[0:2] - self.bin_xy)) < 0.05,
            timeout_s=10.0)
        self.weld_grasp(False)
        # STABLE RELEASE (Hana): open at HALF the grip rate while
        # station-holding with filtered trim, and only leave once calm
        # -- the payload feed-forward is already withdrawn at weld-off,
        # so the thrust step is gone; this removes the jaw-flick and
        # any drift during the open
        ex.set_goal(grip=1.0)
        rel_base = self.data.qpos[0:2].copy()
        rel_z = float(self.data.qpos[2])
        gs0 = A.GRIP_STEP
        A.GRIP_STEP = gs0 * 0.5
        held_r = [0]

        def released_calm():
            ok_ = (ex.grip > 0.99
                   and float(np.linalg.norm(self.data.qvel[0:3])) < 0.06
                   and float(np.linalg.norm(self.data.qvel[3:6])) < 0.15)
            held_r[0] = held_r[0] + 1 if ok_ else 0
            return held_r[0] >= 3
        try:
            self.hold_xy_v2(frames, task, rel_base, rel_z,
                            released_calm, timeout_s=5.0)
        finally:
            A.GRIP_STEP = gs0
        self._grip_contact_ok = False    # grasp phase over
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
            return held[0] >= 3
        self.hold_xy_until(frames, task, here, safe_z, recovered,
                           timeout_s=8.0)
        adr = self.cur["adr"]
        oxy = self.data.qpos[adr:adr + 2]
        return float(np.linalg.norm(oxy - self.bin_xy))

    def episode_v2(self, corrective=False, obj=None, terminal=False,
                   layout=None):
        obj, start, alt, tgt = self.reset_scene_v2(obj=obj,
                                                   corrective=corrective,
                                                   layout=layout)
        task = A.PROMPT_MANIP.format(obj=obj)
        frames = []
        grasped = self.pick_v2(frames, task, alt, corrective=corrective,
                               terminal=terminal)
        d_bin = self.place_v2(frames, task) if grasped else float("nan")
        return frames, dict(obj=obj, corrective=corrective,
                            terminal=terminal,
                            grasped=grasped, d_bin_mm=round(1000 * d_bin, 1)
                            if grasped else None, ticks=len(frames),
                            table_hits=self._table_hits,
                            obj_hits=self._obj_hits,
                            clean=(self._table_hits == 0
                                   and self._obj_hits == 0))


    # -- v2 navigation -----------------------------------------------------
    def nav_v2(self, frames, task):
        """Gate crossing then hover over the named object, in the
        approved leg style: yaw in place, nose-first legs, no pauses
        except the hover that IS the success criterion."""
        ex = self.expert
        gx = self._gate_x
        # crossing height: gripper at least an ARM LENGTH above the
        # gate's bottom member (Hana) -- at 0.95 the jaws ride ~0.45 m
        # over it, and ~0.5 m below the top member
        cross_z = 0.95
        p1 = np.array([gx, -1.05, cross_z])
        self.yaw_then_go(frames, task, p1[0:2])
        ex.set_goal(xyz=p1)
        self.run_until(frames, task,
                       lambda: float(np.linalg.norm(
                           self.data.qpos[0:3] - p1)) < 0.08,
                       timeout_s=14.0)
        p2 = np.array([gx, -0.10, cross_z])
        self.yaw_then_go(frames, task, p2[0:2])
        ex.set_goal(xyz=p2)
        self.run_until(frames, task,
                       lambda: float(self.data.qpos[1]) > -0.12,
                       timeout_s=12.0)
        adr = self.cur["adr"]
        oxy = self.data.qpos[adr:adr + 2].copy()
        self.yaw_then_go(frames, task, oxy)
        hover = np.array([oxy[0], oxy[1], A.PLATE_TOP + 0.40])
        ex.set_goal(xyz=hover)
        self.run_until(frames, task,
                       lambda: float(np.linalg.norm(
                           self.data.qpos[0:2] - oxy)) < 0.10,
                       timeout_s=16.0)
        held = [0]

        def hovered():
            near = float(np.linalg.norm(
                self.data.qpos[0:2]
                - self.data.qpos[adr:adr + 2])) < 0.25
            held[0] = held[0] + 1 if near else 0
            return held[0] >= 35
        return self.hold_xy_v2(frames, task, oxy, hover[2], hovered,
                               timeout_s=10.0)

    def episode_nav(self, obj=None):
        obj, start, alt, tgt = self.reset_scene_v2(obj=obj, nav=True)
        task = A.PROMPT_NAV.format(obj=obj)
        frames = []
        hover_ok = self.nav_v2(frames, task)
        S = np.stack([f["observation.state"] for f in frames])
        gx = self._gate_x
        crossed = False
        for k in range(1, len(S)):
            if (S[k - 1, 1] < -0.6 <= S[k, 1]
                    and abs(float(S[k, 0]) - gx) < 0.45
                    and 0.38 < float(S[k, 2]) < 1.44):
                crossed = True
                break
        return frames, dict(obj=obj, tgt=[round(float(v), 2)
                                          for v in tgt], crossed=crossed,
                            hover=hover_ok,
                            gate_hits=self._gate_hits,
                            table_hits=self._table_hits,
                            obj_hits=self._obj_hits,
                            success=bool(crossed and hover_ok),
                            ticks=len(frames),
                            clean=(self._gate_hits == 0
                                   and self._table_hits == 0
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


def _make_v2_dataset(repo_id):
    """Shared writer setup for every v2-family collection: installs the
    version-proof ffmpeg concat patch (atomic, truncation-guarded) and
    creates the LeRobotDataset with the v2 feature schema. Extracted
    verbatim from collect_main (2026-09-06) so the E3 collection cannot
    drift from the 600-episode-proven path."""
    import subprocess
    import tempfile
    from pathlib import Path
    import imageio_ffmpeg
    import lerobot.datasets.dataset_writer as DW
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def ffmpeg_concat(input_video_paths, output_video_path,
                      overwrite=True, compatibility_check=False):
        # lerobot's pyav concat path needs av>=15, which Myriad's glibc
        # 2.17 ceiling cannot install (av 14.2: the time_base setter
        # raises; av 13.1: canonical_name missing -- both measured).
        # The ffmpeg concat demuxer with stream copy is the same
        # operation, version-proof.
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as fh:
            for p in input_video_paths:
                fh.write("file '%s'\n" % str(Path(p).resolve()))
            lst = fh.name
        out = str(output_video_path)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        # NEVER write onto an input: appends pass output==input[0], and
        # stream-copying onto the file being read truncates it silently
        # (measured: 440 episodes of video reduced to 100 KB). Write to
        # a temp sibling, verify non-trivial, atomically replace.
        tmp = out + ".concat.tmp.mp4"
        r = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0",
                            "-i", lst, "-c", "copy", tmp],
                           capture_output=True)
        Path(lst).unlink(missing_ok=True)
        if r.returncode != 0 or not Path(tmp).exists()                 or Path(tmp).stat().st_size < max(
                    1024, sum(Path(p).stat().st_size
                              for p in input_video_paths) // 2):
            Path(tmp).unlink(missing_ok=True)
            raise RuntimeError("ffmpeg concat failed/truncated: %s"
                               % r.stderr[-400:])
        import os as _os
        _os.replace(tmp, out)

    DW.concatenate_video_files = ffmpeg_concat
    print("FFMPEG-CONCAT-PATCH-ACTIVE", flush=True)
    features = {f"observation.images.{k}": {
        "dtype": "video", "shape": (A.IMG, A.IMG, 3),
        "names": ["height", "width", "channels"]} for k in A.CAMS}
    features["observation.state"] = {"dtype": "float32",
                                     "shape": (len(A.STATE_NAMES),),
                                     "names": A.STATE_NAMES}
    features["scene_state"] = {"dtype": "float32",
                               "shape": (len(A.SCENE_NAMES),),
                               "names": A.SCENE_NAMES}
    features["action"] = {"dtype": "float32",
                          "shape": (len(A.ACTION_NAMES),),
                          "names": A.ACTION_NAMES}
    enc = A.C.pick_rgb_encoder()
    enc.crf = 20
    return LeRobotDataset.create(repo_id=repo_id, fps=A.FPS,
                                 features=features,
                                 robot_type="skygrip",
                                 rgb_encoder=enc)


def _bank_episode(dataset, frames):
    """Write one accepted episode; True on success. A writer failure
    must not kill a long job (the av crash of job 257126)."""
    try:
        for f in frames:
            dataset.add_frame(f)
        dataset.save_episode(parallel_encoding=False)
        return True
    except Exception as e:                # noqa: BLE001
        print("WRITER-ERROR: %s" % e, flush=True)
        return False


def collect_main(repo_id, seed, n_units):
    """The v2 collection: n_units x [9 standard picks + 3 corrective +
    8 nav] = balanced 27:9:24 mix (30 units = 600 episodes: 270/90/240,
    Hana's composition 360 pick-and-place + 240 nav). Every attempt --
    banked or rejected -- is recorded in a jsonl manifest with its
    kind, object, scene configuration, gate counters and outcome, per
    the dissertation checklist (sections B and F)."""
    import json
    import time
    dataset = _make_v2_dataset(repo_id)
    r = V2Runner(seed)
    unit = (["std"] * 9 + ["corr"] * 3 + ["nav"] * 8)
    plan = unit * n_units
    man = open("v2_manifest_%d.jsonl" % seed, "a")
    saved = attempts = 0
    t0 = time.time()
    while saved < len(plan) and attempts < len(plan) * 4:
        kind = plan[saved]
        attempts += 1
        rec = dict(attempt=attempts, slot=saved, kind=kind,
                   collector_seed=seed)
        try:
            if kind == "nav":
                frames, res = r.episode_nav()
                ok = bool(res["success"] and res["clean"])
                reason = (None if ok else
                          "gate-strike" if res["gate_hits"] else
                          "table-clip" if res["table_hits"] else
                          "object-strike" if res["obj_hits"] else
                          "not-crossed" if not res["crossed"] else
                          "no-hover")
            else:
                frames, res = r.episode_v2(corrective=(kind == "corr"))
                placed = (res["d_bin_mm"] is not None
                          and res["d_bin_mm"] <= 150.0)
                ok = bool(res["grasped"] and placed and res["clean"])
                reason = (None if ok else
                          "table-clip" if res["table_hits"] else
                          "object-strike" if res["obj_hits"] else
                          "grasp-miss" if not res["grasped"] else
                          "placed-outside")
        except Exception as e:            # noqa: BLE001 -- log + retry
            rec.update(banked=False, reason="exception:%s" % e)
            man.write(json.dumps(rec) + "\n")
            man.flush()
            continue
        rec.update(res)
        rec.update(banked=ok, reason=reason,
                   parking=parking_window(frames))
        man.write(json.dumps(rec) + "\n")
        man.flush()
        if not ok:
            print("attempt %d [%s] REJECTED: %s" % (attempts, kind,
                                                    reason), flush=True)
            continue
        if not _bank_episode(dataset, frames):
            rec.update(banked=False, reason="writer-error")
            man.write(json.dumps(rec) + "\n")
            man.flush()
            continue
        saved += 1
        if saved % 10 == 0:
            el = time.time() - t0
            print("BANKED %d/%d (%d attempts, %.1f h elapsed, "
                  "eta %.1f h)" % (saved, len(plan), attempts,
                                   el / 3600,
                                   el / 3600 * (len(plan) - saved)
                                   / max(1, saved)), flush=True)
    print("COLLECT-DONE %d/%d banked in %d attempts"
          % (saved, len(plan), attempts), flush=True)


def _judge_pick(res):
    """Shared accept/reject verdict for pick-family episodes (same
    rules as collect_main): grasped + placed inside the box + clean."""
    placed = (res["d_bin_mm"] is not None and res["d_bin_mm"] <= 150.0)
    ok = bool(res["grasped"] and placed and res["clean"])
    reason = (None if ok else
              "table-clip" if res["table_hits"] else
              "object-strike" if res["obj_hits"] else
              "grasp-miss" if not res["grasped"] else
              "placed-outside")
    return ok, reason


def collect_e3_main(repo_id, seed, n_term=150, n_pair=80):
    """The E3 collection (pre-registered; Hana's demo sign-off
    2026-09-06): n_term terminal-corrective episodes (drift-then-yaw-
    correct) + n_pair PAIRED-COMMAND units. A pair = two episodes on
    an identical layout (positions, spawn, box side) differing only in
    the commanded object -- the grounding contrast. Each slot retries
    up to 6 attempts; a pair member retries on ITS OWN layout (member
    A defines it). If B never banks, A stands alone and the pair is
    recorded incomplete. Manifest: e3_manifest_<seed>.jsonl."""
    import json
    import time
    dataset = _make_v2_dataset(repo_id)
    r = V2Runner(seed)
    man = open("e3_manifest_%d.jsonl" % seed, "a")
    saved = attempts = 0
    total = n_term + 2 * n_pair
    t0 = time.time()

    def attempt(kind, slot, pair_id=None, obj=None, layout=None,
                terminal=False):
        nonlocal attempts, saved
        attempts += 1
        rec = dict(attempt=attempts, slot=slot, kind=kind,
                   pair_id=pair_id, collector_seed=seed)
        try:
            frames, res = r.episode_v2(obj=obj, layout=layout,
                                       terminal=terminal)
        except Exception as e:            # noqa: BLE001 -- log + retry
            rec.update(banked=False, reason="exception:%s" % e)
            man.write(json.dumps(rec) + "\n")
            man.flush()
            return False, None
        ok, reason = _judge_pick(res)
        rec.update(res)
        rec.update(banked=ok, reason=reason,
                   parking=parking_window(frames),
                   layout=r.last_layout)
        man.write(json.dumps(rec) + "\n")
        man.flush()
        if not ok:
            print("attempt %d [%s] REJECTED: %s" % (attempts, kind,
                                                    reason), flush=True)
            return False, None
        if not _bank_episode(dataset, frames):
            rec2 = dict(rec, banked=False, reason="writer-error")
            man.write(json.dumps(rec2) + "\n")
            man.flush()
            return False, None
        saved += 1
        if saved % 10 == 0:
            el = time.time() - t0
            print("BANKED %d/%d (%d attempts, %.1f h, eta %.1f h)"
                  % (saved, total, attempts, el / 3600,
                     el / 3600 * (total - saved) / max(1, saved)),
                  flush=True)
        return True, res

    for slot in range(n_term):
        for _ in range(6):
            ok, _res = attempt("term", slot, terminal=True)
            if ok:
                break
    for p in range(n_pair):
        slot = n_term + p
        okA = False
        for _ in range(6):
            okA, resA = attempt("pairA", slot, pair_id=p)
            if okA:
                break
        if not okA:
            continue
        layout = dict(r.last_layout)
        objB = [k for k in r.objs if k != resA["obj"]][0]
        okB = False
        for _ in range(6):
            okB, _resB = attempt("pairB", slot, pair_id=p, obj=objB,
                                 layout=layout)
            if okB:
                break
        if not okB:
            print("pair %d INCOMPLETE (A banked alone)" % p,
                  flush=True)
    print("COLLECT-E3-DONE %d/%d banked in %d attempts"
          % (saved, total, attempts), flush=True)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    assert mode in ("demo", "demonav", "demoterm", "collect",
                    "collecte3")
    if mode == "collecte3":
        # usage: collecte3 <repo_id> <seed> [n_term n_pair]
        collect_e3_main(out, int(sys.argv[3]),
                        int(sys.argv[4]) if len(sys.argv) > 4 else 150,
                        int(sys.argv[5]) if len(sys.argv) > 5 else 80)
        sys.exit(0)
    if mode == "demoterm":
        # E3 terminal-corrective flavour demo (6 eps, coin-flip
        # objects) for sign-off before any collection
        r = V2Runner(34000)
        w = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                               macro_block_size=1)
        for _ in range(6):
            frames, res = r.episode_v2(terminal=True)
            print("TERM %-14s grasped=%-5s d_bin=%s mm  dmin=%.1fmm "
                  "fired=%s table_hits=%d obj_hits=%d %s (%d ticks)"
                  % (res["obj"], res["grasped"], res["d_bin_mm"],
                     1000 * r._creep_diag["dmin"],
                     r._creep_diag["fired"],
                     res["table_hits"], res["obj_hits"],
                     "CLEAN" if res["clean"] else "REJECT",
                     res["ticks"]), flush=True)
            for f in frames[::2]:
                img = np.concatenate(
                    [f["observation.images.camera1"],
                     f["observation.images.camera2"],
                     f["observation.images.camera3"]], axis=1)
                w.append_data(img)
        w.close()
        print("demo ->", out)
        sys.exit(0)
    if mode == "collect":
        collect_main(out, int(sys.argv[3]),
                     int(sys.argv[4]) if len(sys.argv) > 4 else 30)
        sys.exit(0)
    if mode == "demonav":
        # fresh seed, BALANCED targets (Hana: 3 penguin + 3 weight),
        # positions fully randomized across the tabletop
        r = V2Runner(33000)
        w = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                               macro_block_size=1)
        for obj in ("plush penguin", "weight", "plush penguin",
                    "weight", "plush penguin", "weight"):
            frames, res = r.episode_nav(obj=obj)
            print("NAV %-14s tgt=%s crossed=%-5s hover=%-5s gate_hits=%d "
                  "table_hits=%d obj_hits=%d %s (%d ticks)"
                  % (res["obj"], res["tgt"], res["crossed"], res["hover"],
                     res["gate_hits"], res["table_hits"], res["obj_hits"],
                     "CLEAN" if res["clean"] and res["success"]
                     else "REJECT", res["ticks"]), flush=True)
            for f in frames[::2]:
                img = np.concatenate(
                    [f["observation.images.camera1"],
                     f["observation.images.camera2"],
                     f["observation.images.camera3"]], axis=1)
                w.append_data(img)
        w.close()
        print("demo ->", out)
        sys.exit(0)
    r = V2Runner(31000)
    w = imageio.get_writer(out, fps=10, codec="libx264", quality=8,
                           macro_block_size=1)
    # 10-episode demo (Hana): 7 standard + 3 corrective, objects
    # coin-flipped per episode so the scenes vary
    for corrective, obj in ((False, None), (False, None), (False, None),
                            (True, None), (False, None), (False, None),
                            (True, None), (False, None), (False, None),
                            (True, None)):
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
