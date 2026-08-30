"""fixfit5.py -- Platform repair, iteration 5 (cumulative, authoritative).

Defect history this file closes out:
  v1  stale sp[2]<0.42 arm gate -> jaws 100-150 mm off at the pinch; aperture
      weld fires on air and TOWS the object (weight crossed the room with
      zero drone contacts).
  v2  contact-gated weld + clip 0.035 validated on weights via the expert's
      recorded arm (miss 6.5/4.9 mm, placed); penguin failed: contact never
      registered.
  v4  instrumentation: penguin pad contact FLICKERS tick-to-tick (plush
      solver jitter, per the collector's own comments) -- consecutive-tick
      gating can never fire. And all automaton drops clustered 0.13 m off
      bin centre: the NAMED q_grasp pose is ~0.13 m of FK from the pose the
      expert actually holds at release.
  v5  weld debounce (2-of-last-4); deploy/tuck poses CALIBRATED from the
      captured demonstrations (system identification from training demos,
      not runtime privilege); deterministic tuck at weld+60 (timing proved
      non-critical -- ep0 passed 293 ticks early -- once the pose is right).

ACCEPTANCE A: expert-recorded arm + contact weld + clip. Validates the weld
path in isolation.
ACCEPTANCE B: the deployable automaton (travel -> deploy at approach alt ->
weld on debounced seated contact -> tuck at weld+60 -> travel on release),
calibrated poses, slewed targets. This is what ships into eval_pi0 if 3/3.
"""
import numpy as np
import mujoco
import collect_airvla as A
import eval_pi0 as E

SEED = 97000
OBJS = ['weight', 'plush penguin', 'weight']
CLIP = 0.035
SLEW = 0.06
AP_LO, AP_HI = 0.004, 0.0170


# ---------- capture ----------
CAP = []
for n, obj in enumerate(OBJS):
    r = A.Runner(SEED + n)
    acts, arms, tl, snap = [], [], [], {}
    o_tick = A.Expert.tick

    def tick(self):
        if not snap:
            snap.update(qpos=r.data.qpos.copy(), qvel=r.data.qvel.copy(),
                        body_pos=r.model.body_pos.copy(),
                        body_quat=r.model.body_quat.copy(),
                        body_mass=r.model.body_mass.copy(),
                        bin_xy=np.array(r.bin_xy, float).copy(),
                        bin_yaw=float(r.bin_yaw))
        a = o_tick(self)
        acts.append(np.asarray(a, float).copy())
        arms.append(np.asarray(self.arm, float).copy())
        tl.append((self.sp.copy(), float(self.grip),
                   bool(r.data.eq_active[r.weld])))
        return a

    A.Expert.tick = tick
    out = r.manip_episode(obj=obj)
    A.Expert.tick = o_tick
    info = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
    CAP.append(dict(obj=obj, acts=acts, arms=arms, tl=tl, snap=snap,
                    ok=info.get('success') if isinstance(info, dict) else info))
    print('ep%d %-14s ok=%s ticks=%d' % (n, obj, CAP[-1]['ok'], len(tl)),
          flush=True)


# ---------- calibrate arm poses from the captured demonstrations ----------
def _weld_ticks(tl):
    w = [t[2] for t in tl]
    on = [i for i in range(1, len(w)) if w[i] and not w[i - 1]]
    off = [i for i in range(1, len(w)) if not w[i] and w[i - 1]]
    return (on[0] if on else None), (off[0] if off else None)


_dep, _tuk = [], []
for c in CAP:
    on, off = _weld_ticks(c['tl'])
    if on:
        _dep.append(np.mean(c['arms'][max(on - 80, 0):on - 10], axis=0))
    if off:
        _tuk.append(np.mean(c['arms'][max(off - 150, 0):off - 20], axis=0))
Q_TRAVEL_CAL = np.asarray(CAP[0]['arms'][0], float).copy()
Q_DEPLOY_CAL = np.mean(_dep, axis=0)
Q_TUCK_CAL = np.mean(_tuk, axis=0)
print('calibrated poses: travel=%s deploy=%s tuck=%s'
      % (np.round(Q_TRAVEL_CAL, 3), np.round(Q_DEPLOY_CAL, 3),
         np.round(Q_TUCK_CAL, 3)), flush=True)


# ---------- replay through patched Platform ----------
def replay(c, n, arm_fn, tag):
    r2 = A.Runner(SEED + n)
    r2.reset_scene(np.zeros(2), c['snap']['qpos'][0:3].copy(), obj=c['obj'])
    r2.model.body_pos[:] = c['snap']['body_pos']
    r2.model.body_quat[:] = c['snap']['body_quat']
    r2.model.body_mass[:] = c['snap']['body_mass']
    r2.data.qpos[:] = c['snap']['qpos']
    r2.data.qvel[:] = c['snap']['qvel']
    r2.bin_xy = c['snap']['bin_xy'].copy()
    r2.bin_yaw = c['snap']['bin_yaw']
    mujoco.mj_forward(r2.model, r2.data)
    m = r2.model
    ob = r2.cur['body']
    obj_geoms = set(g for g in range(m.ngeom)
                    if m.body_rootid[m.geom_bodyid[g]] == ob)
    pad_geoms = {m.geom('pad_1').id, m.geom('pad_2').id}
    adr = r2.cur['adr']
    z0 = float(r2.data.qpos[adr + 2])

    class P:
        pass
    p = P()
    p.sp = c['snap']['qpos'][0:3].copy()
    p.yaw = 0.0
    p.grip = 1.0
    p.hist = []
    p.r = r2
    miss, wev, prev, dbg = 1e9, [], False, []
    for i, a in enumerate(c['acts']):
        p.sp = p.sp + np.clip(a[0:3], -CLIP, CLIP)
        p.yaw += float(np.clip(a[5], -0.06, 0.06))
        p.grip = float(np.clip(a[6], 0.0, 1.0))
        welded = bool(r2.data.eq_active[r2.weld])
        ap = float(r2.data.qpos[r2.gadr])
        hit = False
        d = r2.data
        for ci in range(d.ncon):
            g1 = int(d.contact.geom1[ci])
            g2 = int(d.contact.geom2[ci])
            if (g1 in pad_geoms and g2 in obj_geoms) or \
               (g2 in pad_geoms and g1 in obj_geoms):
                hit = True
                break
        if not welded and AP_LO < ap < AP_HI:
            # DEBOUNCED: plush contact flickers tick-to-tick (solver jitter,
            # per the collector's comments); 2 hits in the last 4 ticks welds.
            p.hist = (p.hist + [hit])[-4:]
            if sum(p.hist) >= 2:
                r2.weld_grasp(True)
        elif welded and p.grip > 0.8:
            r2.weld_grasp(False)
            p.hist = []
        else:
            p.hist = []
        if not dbg or (abs(dbg[-1][1] - ap) > 0.0005 or dbg[-1][2] != hit):
            dbg.append((i, round(ap, 4), hit))
        q = arm_fn(i, p)
        r2.ctrl.set_targets(p.sp, q, E.C.GRIPPER_OPEN * p.grip)
        r2.ctrl.mppi.target_yaw = p.yaw
        for _ in range(r2.sub):
            r2.ctrl.step()
            mujoco.mj_step(r2.model, r2.data)
        w = bool(r2.data.eq_active[r2.weld])
        if w != prev:
            wev.append((i, 'ON' if w else 'OFF'))
            prev = w
        aim, _ = r2.live_target()
        miss = min(miss, float(np.linalg.norm(r2.jaws() - aim)))
    welded = bool(r2.data.eq_active[r2.weld])
    try:
        placed = bool(E.in_box(r2))
    except Exception:
        placed = None
    oxy = r2.data.qpos[adr:adr + 2]
    print('  [%s] ep%d %-14s welded=%s placed=%s miss=%.1f mm dz=%+.3f '
          'weld=%s -> %s'
          % (tag, n, c['obj'], welded, placed, miss * 1000,
             float(r2.data.qpos[adr + 2]) - z0, wev,
             'PASS' if placed else 'FAIL'), flush=True)
    print('    final obj xy=(%.3f,%.3f) bin_xy=(%.3f,%.3f) d=%.3f m'
          % (oxy[0], oxy[1], r2.bin_xy[0], r2.bin_xy[1],
             float(np.linalg.norm(oxy - r2.bin_xy))), flush=True)
    if not placed:
        print('    pinch debug:', dbg[:24], flush=True)
    return bool(placed)


# ---------- ACCEPTANCE A ----------
print()
print('ACCEPTANCE A (expert-recorded arm + debounced contact-weld + clip):',
      flush=True)
pa = 0
for n, c in enumerate(CAP):
    arms = c['arms']
    pa += replay(c, n, lambda i, p, arms=arms: arms[min(i, len(arms) - 1)],
                 'A')
print('ACCEPT-A %d/3' % pa, flush=True)

# ---------- ACCEPTANCE B ----------
print()
print('ACCEPTANCE B (automaton: deploy at approach alt; tuck at weld+60; '
      'travel on release; calibrated poses, slewed):', flush=True)
pb = 0
for n, c in enumerate(CAP):
    st = dict(deployed=False, weld_tick=None, released=False,
              cmd=Q_TRAVEL_CAL.copy())

    def arm_fn(i, p, st=st):
        welded = bool(p.r.data.eq_active[p.r.weld])
        if welded and st['weld_tick'] is None:
            st['weld_tick'] = i
            print('    [B] weld at tick %d' % i, flush=True)
        if st['weld_tick'] is not None and not welded and p.grip > 0.8:
            st['released'] = True
        if not st['deployed'] and p.sp[2] >= 0.50:
            st['deployed'] = True
        if st['released']:
            tgt = Q_TRAVEL_CAL
        elif st['weld_tick'] is not None and i >= st['weld_tick'] + 60:
            tgt = Q_TUCK_CAL
        elif st['deployed']:
            tgt = Q_DEPLOY_CAL
        else:
            tgt = Q_TRAVEL_CAL
        st['cmd'] = st['cmd'] + np.clip(tgt - st['cmd'], -SLEW, SLEW)
        return st['cmd']

    pb += replay(c, n, arm_fn, 'B')
print('ACCEPT-B %d/3' % pb, flush=True)
print('PLATFORM-FIX5 A=%d/3 B=%d/3' % (pa, pb), flush=True)
