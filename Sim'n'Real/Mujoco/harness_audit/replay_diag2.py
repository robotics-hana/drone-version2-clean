"""replay_diag2.py -- ground-truth replay, SCENE-FAITHFUL version.

v1 was contaminated: it re-called reset_scene(captured args) on a fresh
Runner, but reset_scene draws ~11 values from self.rng internally (object
yaw, distractor offset, BIN POSITION, bin yaw, bin2, tint), and the fresh
rng had not consumed manip_episode's earlier draws -- so the replay scene
differed from the collection scene and the expert's captured actions were
flown against the wrong world. v1's 116-151 mm miss proves nothing.

v2 snapshots the COMPLETE initial state at the first expert tick of the
collection episode -- d.qpos, d.qvel, m.body_pos, m.body_quat, m.body_mass,
plus Runner.bin_xy/bin_yaw -- and restores it before the replay, so the
replay world is byte-identical and the only variable is Platform itself.

Also instruments:
  WELD timeline -- tick of every eq_active transition (the air-weld tow is
  the only mechanism that fits an object crossing the room with no contact)
  ARM dwell, slew-aware -- per tick, nearest of {q_travel, q_grasp, q_carry}
  contacts on the task object + first-contact tick vs gate crossings
  miss via live_target(), the harness's own metric
"""
import sys, itertools
import numpy as np
import mujoco
import collect_airvla as A
import eval_pi0 as E

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 97000
GATE = 0.42
OBJS = ['weight', 'plush penguin', 'weight']

def gname(m, i):
    n = m.geom(i).name
    return n if n else 'geom%d' % i

def arm_label(a, ex):
    cands = []
    for nm in ('q_travel', 'q_grasp', 'q_carry'):
        q = getattr(ex, nm, None)
        if q is not None:
            cands.append((float(np.linalg.norm(np.asarray(a, float)
                                               - np.asarray(q, float))), nm))
    return min(cands)[1] if cands else '?'

for n, obj in enumerate(OBJS):
    r = A.Runner(SEED + n)
    acts, arms, snap = [], [], {}
    o_tick = A.Expert.tick
    def tick(self):
        if not snap:                       # first tick: freeze the world
            snap['qpos'] = r.data.qpos.copy()
            snap['qvel'] = r.data.qvel.copy()
            snap['body_pos'] = r.model.body_pos.copy()
            snap['body_quat'] = r.model.body_quat.copy()
            snap['body_mass'] = r.model.body_mass.copy()
            snap['bin_xy'] = np.array(r.bin_xy, float).copy()
            snap['bin_yaw'] = float(r.bin_yaw)
            snap['obj'] = obj
        a = o_tick(self)
        acts.append(np.asarray(a, float).copy())
        arms.append(np.asarray(self.arm, float).copy())
        return a
    A.Expert.tick = tick
    out = r.manip_episode(obj=obj)
    A.Expert.tick = o_tick
    info = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
    ok = info.get('success') if isinstance(info, dict) else info
    lab = [arm_label(a, r.expert) for a in arms]
    eg = sum(1 for x in lab if x == 'q_grasp')
    ec = sum(1 for x in lab if x == 'q_carry')
    print('ep%d %-14s EXPERT success=%s ticks=%d  arm dwell: grasp=%d carry=%d'
          % (n, obj, ok, len(acts), eg, ec), flush=True)

    # ---- faithful replay ----
    r2 = A.Runner(SEED + n)
    r2.reset_scene(np.zeros(2), snap['qpos'][0:3].copy(), obj=obj)  # structures only
    r2.model.body_pos[:] = snap['body_pos']
    r2.model.body_quat[:] = snap['body_quat']
    r2.model.body_mass[:] = snap['body_mass']
    r2.data.qpos[:] = snap['qpos']
    r2.data.qvel[:] = snap['qvel']
    r2.bin_xy = snap['bin_xy'].copy()
    r2.bin_yaw = snap['bin_yaw']
    mujoco.mj_forward(r2.model, r2.data)
    start = snap['qpos'][0:3].copy()
    plat = E.Platform(r2, start, 0.0)
    m = r2.model
    ob = r2.cur['body']
    obj_geoms = set(range(m.body_geomadr[ob], m.body_geomadr[ob] + m.body_geomnum[ob]))
    adr = r2.cur['adr']; z0 = float(r2.data.qpos[adr + 2])
    spz, touch, miss = [], {}, 1e9
    first_contact = None; weld_events = []; prev_weld = False
    for a in acts:
        plat.tick(a)
        spz.append(float(plat.sp[2]))
        w = bool(r2.data.eq_active[r2.weld])
        if w != prev_weld:
            weld_events.append((len(spz) - 1, 'ON' if w else 'OFF'))
            prev_weld = w
        aim_now, _ = r2.live_target()
        miss = min(miss, float(np.linalg.norm(r2.jaws() - aim_now)))
        d = r2.data
        for ci in range(d.ncon):
            g1, g2 = int(d.contact.geom1[ci]), int(d.contact.geom2[ci])
            if g1 in obj_geoms or g2 in obj_geoms:
                other = g2 if g1 in obj_geoms else g1
                k = gname(m, other)
                touch[k] = touch.get(k, 0) + 1
                if first_contact is None and not k.startswith(('table', 'ground', 'stand')):
                    first_contact = len(spz) - 1
    spz = np.array(spz); below = spz < GATE
    xs = np.where(below[1:] != below[:-1])[0]
    longest = max((sum(1 for _ in g) for k, g in itertools.groupby(below) if k), default=0)
    welded = bool(r2.data.eq_active[r2.weld])
    lifted = float(r2.data.qpos[adr + 2]) - z0 > 0.03
    try:
        placed = bool(E.in_box(r2))
    except Exception:
        placed = None
    print('   REPLAY  welded=%s lifted=%s placed=%s  MISS(live_target)=%.1f mm  obj dz=%+.3f'
          % (welded, lifted, placed, miss * 1000,
             float(r2.data.qpos[adr + 2]) - z0), flush=True)
    print('   WELD events: %s' % (weld_events if weld_events else 'none'), flush=True)
    print('   AUTOMATON  ticks<gate=%d crossings=%d longest=%d min_spz=%.3f | crossing ticks=%s'
          % (int(below.sum()), len(xs), longest, spz.min(), [int(x) for x in xs[:8]]), flush=True)
    print('   CONTACTS: %s | first non-rest contact tick=%s'
          % (sorted(touch.items(), key=lambda kv: -kv[1])[:8], first_contact), flush=True)
    print('   drone end=%s' % np.round(r2.data.qpos[0:3], 3), flush=True)
