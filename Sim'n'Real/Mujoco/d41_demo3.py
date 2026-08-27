"""Simple-carry verification demo (user spec 2026-08-26): 2 weight +
2 penguin production episodes with per-tick traces and QUANTITATIVE
acceptance gates for the simplified profile:
  (1) no dive below target altitude on the approach,
  (2) no fast lunge near the stand,
  (3) yaw-in-place: payload offset RIGID IN THE BODY FRAME during the
      turn (the arm stays extended, so the payload must rotate WITH the
      nose -- any body-frame excursion would be swing or orbit),
  (4) zero arm motion from grasp to release (no tuck, no re-extension).
"""
import imageio.v2 as imageio
import numpy as np
import collect_airvla as A

r = A.Runner(97000)
vid = []
trace = []
LEG_IDS = {A.mujoco.mj_name2id(r.model, A.mujoco.mjtObj.mjOBJ_GEOM, n)
           for n in ('leg_front_left', 'leg_front_right',
                     'leg_back_left', 'leg_back_right')}
TAB_ID = A.mujoco.mj_name2id(r.model, A.mujoco.mjtObj.mjOBJ_GEOM,
                             'table_top')
legc = [0]
orig_step = A.Runner.step


def stepv(self, frames, task):
    orig_step(self, frames, task)
    for ci in range(self.data.ncon):
        c1 = int(self.data.contact.geom1[ci])
        c2 = int(self.data.contact.geom2[ci])
        if TAB_ID in (c1, c2) and (c1 in LEG_IDS or c2 in LEG_IDS):
            legc[0] += 1
    ex = self.expert
    adr = self.cur['adr']
    trace.append([
        float(self.data.qpos[0]), float(self.data.qpos[1]),
        float(self.data.qpos[2]),
        float(self.data.qpos[adr]), float(self.data.qpos[adr + 1]),
        float(self.data.qpos[adr + 2]),
        float(np.linalg.norm(self.data.qvel[0:2])),
        float(ex.grip), float(ex.yaw), float(ex.goal_yaw),
        float(self.data.qpos[7]), float(self.data.qpos[8])])
    if len(vid) < 6000 and len(frames) % 2 == 0:
        t = []
        for cam in (A.CAMS['camera1'], A.CAMS['camera2'], 'lab_external'):
            self.rend.update_scene(self.data, camera=cam)
            t.append(self.rend.render().copy())
        vid.append(np.concatenate(t, axis=1))


A.Runner.step = stepv

all_ok = True
EPS = [('weight', {}), ('plush penguin', {}), ('weight', {}),
       ('plush penguin', {}),
       # corrective flavour was exercised by NO gate (workflow review)
       ('plush penguin', {'fallen_start': True})]
for i, (obj, kw) in enumerate(EPS):
    trace.clear()
    legc[0] = 0
    frames, info = r.manip_episode(obj=obj, **kw)
    T = np.array(trace)
    bz, oz = T[:, 2], T[:, 5]
    sxy = T[0, 3:5]                       # stand/object spawn xy
    dstand = np.linalg.norm(T[:, 0:2] - sxy, axis=1)
    grip = T[:, 7]
    # approach window ends at the FINAL close (user 2026-08-26: the
    # first-close cutoff let retry flights dodge every approach gate)
    below = grip < 0.9
    starts = np.where(below[1:] & ~below[:-1])[0] + 1
    if len(below) and below[0]:
        starts = np.concatenate([[0], starts])
    ic = int(starts[-1]) if len(starts) else len(T)
    # (1) approach dive: min body z vs z at close, measured only AFTER
    # grasp altitude is first reached -- the episode now deliberately
    # opens BELOW it (up-then-across takeoff, user 2026-08-26)
    alt_close = bz[max(0, ic - 5):ic + 1].mean() if ic < len(T) else bz[-1]
    reach = np.where(bz[:ic] >= alt_close - 0.02)[0]
    dive = float(alt_close - bz[reach[0]:ic].min()) if len(reach) else 0.0
    # (1b) approach SINK RATE: max 1-second altitude loss anywhere on
    # the approach. The dive gate misses a plunge that STOPS at grasp
    # altitude (user review 2026-08-26: high spawn near the stand ->
    # 0.18 m/s vertical drop beside the pole); at the tightened 7-deg
    # spawn slope cap a flown glide sinks <= ~0.04 m/s.
    sink = float((bz[:ic - 10] - bz[10:ic]).max()) if ic > 10 else 0.0
    # (1c) STABLE HOVER when the arm reaches out: horizontal speed at
    # the tick the arm joints start moving toward the grasp pose
    dq = np.abs(np.diff(T[:ic, 10:12], axis=0)).sum(axis=1) \
        if ic > 1 else np.zeros(1)
    moving = np.where(dq > 0.004)[0]
    extv = float(T[moving[0], 6]) if len(moving) else 0.0
    # (2) fast lunge near the stand: max speed within 0.30 m of it
    near = dstand[:ic] < 0.30
    lunge = float(T[:ic][near, 6].max()) if near.any() else 0.0
    # (3) yaw-in-place: payload offset in the BODY frame during the
    # post-grasp turn. Extended-arm carry means the world-frame offset
    # ROTATES with the nose by design; rigidity in the body frame is
    # the true "turns about its own axis, payload just rides along".
    # no altitude floor (workflow review: the loaded droop could sink
    # the whole turn under a bz>0.85 floor, emptying the window and
    # passing the -1 sentinels): during the carry the ONLY commanded
    # yaw activity is the bin turn, so grip+yaw-error alone window it
    hold = grip < 0.9
    turning = hold & (np.abs(T[:, 8] - T[:, 9]) > 0.05)
    if turning.any():
        rel = T[turning][:, 3:5] - T[turning][:, 0:2]
        yw = T[turning][:, 8]
        c, s = np.cos(-yw), np.sin(-yw)
        rb = np.stack([c * rel[:, 0] - s * rel[:, 1],
                       s * rel[:, 0] + c * rel[:, 1]], axis=1)
        relarc = float(np.linalg.norm(rb - rb.mean(axis=0),
                                      axis=1).max())
        wdrift = float(np.linalg.norm(
            T[turning][:, 0:2] - T[turning][:, 0:2].mean(axis=0),
            axis=1).max())
    else:
        relarc, wdrift = -1.0, -1.0
    # (3b) box approach must not overshoot (user 2026-08-26: the lead
    # compensation regression sailed the payload past the bin and
    # walked it back): carried payload never passes the bin centre by
    # more than 10 cm along the stand->bin carry direction
    # no altitude floor here either (workflow review: the drooped
    # arrival ticks -- exactly where an overshoot lives -- fell under
    # the old bz>0.70 mask)
    bxy_bin = np.array(r.bin_xy, float)
    cw = (grip < 0.9) & (np.abs(T[:, 8] - T[:, 9]) < 0.05)
    if cw.any():
        u_c = bxy_bin - sxy
        u_c = u_c / max(1e-9, float(np.linalg.norm(u_c)))
        over = float(((T[cw][:, 3:5] - bxy_bin) @ u_c).max())
    else:
        over = 0.0
    # (3c) carry LEVEL: from the top of the climb until the payload
    # nears the bin, altitude must hold (workflow review: the old
    # cruise shed 0.25-0.31 m en route -- a long shallow dive at the
    # box); the deliberate descent over the bin is excluded
    carried = np.where(grip < 0.9)[0]
    zdrop = 0.0
    if len(carried) > 20:
        t_top = carried[int(np.argmax(bz[carried]))]
        dbin = np.linalg.norm(T[:, 3:5] - bxy_bin, axis=1)
        seg = [t for t in carried if t >= t_top and dbin[t] > 0.25]
        if len(seg) > 5:
            zdrop = float(bz[t_top] - bz[seg].min())
    # (3d) shared-table clearances (table era, 2026-08-27): keep
    # 28 cm off the DISTRACTOR object while low, and keep the leg
    # tips (11.6 cm below the body at 4/5 length) above the tabletop
    # (z 0.390) whenever the body is over the footprint + 17 cm
    # reach margin -- the scrape is the new failure mode
    other = 'weight' if obj != 'weight' else 'plush penguin'
    oadr = r.objs[other]['adr']
    dxy_o = np.array(r.data.qpos[oadr:oadr + 2], float)
    low = bz < 0.70
    if low.any():
        s2min = float(np.linalg.norm(T[low][:, 0:2] - dxy_o,
                                     axis=1).min())
    else:
        s2min = 9.9
    tc = np.array(r.model.body_pos[r.table_id][0:2], float)
    overt = ((np.abs(T[:, 0] - tc[0]) < 0.62)
             & (np.abs(T[:, 1] - tc[1]) < 0.47))
    scrape = float(bz[overt].min()) if overt.any() else 9.9
    # (3e) grasp/release stability (user 2026-08-26: "grasp then
    # hover, not grasp and swing forward"; same at the drop): max
    # horizontal body speed in the ~1.2 s after the final close and
    # after the release must read as a steady hover
    gspeed = float(T[ic:ic + 12, 6].max()) if ic + 12 <= len(T) else 0.0
    ci = np.where(grip < 0.9)[0]
    rspeed = 0.0
    if len(ci) and ci[-1] + 12 <= len(T):
        rspeed = float(T[ci[-1] + 1:ci[-1] + 13, 6].max())
    # (3f) calm WIND-DOWN (user 2026-08-26: "places calmly ... then
    # suddenly becomes jerky"): from the release to the episode end,
    # every move stays at drift pace
    endv = float(T[ci[-1] + 1:, 6].max()) if len(ci) else 0.0
    # (4) arm frozen while carried, EXCEPT motion segments that BEGIN
    # with the payload over the bin (<0.15 m) -- the release tuck. The
    # tuck's arc legitimately sweeps the welded payload ~0.30 m out
    # mid-fold (probe-measured 0.302 for the weight), so a
    # radius-only exemption misfires at the arc's apex; keying on the
    # segment's START tick keeps the gate sharp everywhere else.
    dbin4 = np.linalg.norm(T[:, 3:5] - bxy_bin, axis=1)
    dq4 = np.concatenate([[0.0], np.abs(
        np.diff(T[:, 10:12], axis=0)).sum(axis=1)])
    mov = (grip < 0.9) & (dq4 > 0.004)
    armmove = 0.0
    i2 = 0
    while i2 < len(T):
        if mov[i2]:
            j2 = i2
            while j2 + 1 < len(T) and mov[j2 + 1]:
                j2 += 1
            if dbin4[i2] > 0.15:
                armmove += float(dq4[i2:j2 + 1].sum())
            i2 = j2 + 1
        else:
            i2 += 1
    g1 = dive <= 0.05
    g2 = lunge <= 0.15
    g3 = relarc <= 0.05
    g4 = 0.0 <= armmove <= 0.06
    # (5) the BODY itself holds station during the turn -- this is the
    # visible "yaw on the exact spot" (the loaded-trim arc measured
    # 0.46 m before the lead-compensated turn went in)
    g5 = wdrift <= 0.15
    g6 = sink <= 0.06
    g7 = extv <= 0.04
    g8 = over <= 0.10
    g9 = zdrop <= 0.10
    g10 = s2min >= 0.28
    g11 = gspeed <= 0.08
    g12 = rspeed <= 0.08
    g13 = endv <= 0.15
    g14 = scrape >= 0.511
    # (3g) physics-verified gentleness: the LEG geoms never contact
    # the tabletop, and the object never moves before the FIRST close
    # command (user 2026-08-27: "gripper pushed too far -- jiggles")
    ic0 = int(starts[0]) if len(starts) else len(T)
    if ic0 > 12:
        o0 = T[10, 3:5]
        prenudge = float(np.linalg.norm(T[10:ic0, 3:5] - o0,
                                        axis=1).max())
    else:
        prenudge = 0.0
    g15 = prenudge <= 0.006
    g16 = legc[0] == 0
    # (3h) the episode ENDS at a pinned hover (user 2026-08-27)
    finalv = float(T[-12:, 6].max())
    g17 = finalv <= 0.05
    ok = info['success'] and info.get('gate', True) and g1 and g2 \
        and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10 \
        and g11 and g12 and g13 and g14 and g15 and g16 and g17
    all_ok &= ok
    print('DEMO %s%s success=%s gate=%s retried=%s | dive=%.3f(%s) '
          'sink=%.2f(%s) extv=%.3f(%s) lunge=%.2f(%s) relarc=%.3f(%s) '
          'bodydrift=%.3f(%s) overshoot=%.3f(%s) carrydrop=%.3f(%s) '
          'objclear=%.2f(%s) scrape=%.2f(%s) nudge=%.3f(%s) '
          'leghit=%d(%s) graspv=%.3f(%s) '
          'dropv=%.3f(%s) endv=%.2f(%s) finalv=%.3f(%s) '
          'armmove=%.3f(%s) frames=%d'
          % (obj, ' [corrective]' if kw else '', info['success'],
             info.get('gate'), info.get('retried'), dive,
             'OK' if g1 else 'FAIL', sink, 'OK' if g6 else 'FAIL',
             extv, 'OK' if g7 else 'FAIL',
             lunge, 'OK' if g2 else 'FAIL',
             relarc, 'OK' if g3 else 'FAIL', wdrift,
             'OK' if g5 else 'FAIL', over, 'OK' if g8 else 'FAIL',
             zdrop, 'OK' if g9 else 'FAIL',
             s2min, 'OK' if g10 else 'FAIL',
             scrape, 'OK' if g14 else 'FAIL',
             prenudge, 'OK' if g15 else 'FAIL',
             legc[0], 'OK' if g16 else 'FAIL',
             gspeed, 'OK' if g11 else 'FAIL',
             rspeed, 'OK' if g12 else 'FAIL',
             endv, 'OK' if g13 else 'FAIL',
             finalv, 'OK' if g17 else 'FAIL',
             armmove, 'OK' if g4 else 'FAIL', len(frames)),
          flush=True)

w = imageio.get_writer('/clusterhome/hana/d41_demo3.mp4', fps=10,
                       codec='libx264', quality=8, macro_block_size=1)
for f in vid:
    w.append_data(f)
w.close()
print('SIMPLE-CARRY-DONE all_gates_pass=%s frames=%d' % (all_ok, len(vid)),
      flush=True)
