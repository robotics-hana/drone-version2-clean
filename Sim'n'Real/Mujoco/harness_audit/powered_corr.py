"""powered_corr.py -- spawn-correlation test at n=66 from the archived
trajectory log, while the n=20 audit run queues.

Episode identification does NOT trust positional counting alone:
  * honest-harness episodes are identified by frame-0 altitude < 0.45
    (--startfix deck spawns; training/eval overlap analysis guarantees the
    broken-harness runs started at z >= 0.55, so the discriminator is exact);
  * rtc- and guided-mode startfix picks are unambiguous outright (each mode
    ran exactly one honest pick block);
  * naive startfix picks are 4 blocks in append order -- both30k(20),
    b60k(20), objyaw(20), vid(6) -- COUNTS ASSERTED to equal 66 exactly, and
    the 30k block and the yaw-pinned block are excluded.
Miss here is body-to-object closest xy (old records lack jaw positions); the
body-to-jaw offset is ~constant, and adding a constant to miss leaves a
correlation unchanged.

Statistic (per the review): Mahalanobis spawn distance (per-axis std of the
225 training pick objects), mode treated as covariate by demeaning miss
within block, plus the directional regressions (world-x miss component vs
world-x object deviation; same for y).
"""
import json
import numpy as np

F = '/home/ucabhe0/Scratch/airvla/evalout/eval_traj_pre_audit.jsonl'
recs = [json.loads(l) for l in open(F)]
train = np.load('/home/ucabhe0/Scratch/airvla/evalout/train_obj_xy.npy')
mu, sd = train.mean(0), train.std(0)
print('training prior: mean (%.3f, %.3f)  std (%.3f, %.3f)'
      % (mu[0], mu[1], sd[0], sd[1]))

picks = [r for r in recs if r['kind'] == 'pick']
sfx = [r for r in picks if np.array(r['traj'])[0, 2] < 0.45]
by_mode = {}
for r in sfx:
    by_mode.setdefault(r['mode'], []).append(r)
counts = {m: len(v) for m, v in by_mode.items()}
print('startfix picks by mode:', counts)
assert counts.get('rtc') == 20, 'rtc block ambiguous'
assert counts.get('guided') == 20, 'guided block ambiguous'
assert counts.get('naive') == 86, 'naive blocks not 86 -- do not trust order'
# append order: startfix30k(20), both30k(20), b60k(20), objyaw(20), vid(6)
nv = by_mode['naive']
blocks = {'b60k': nv[40:60], 'vid': nv[80:86],
          'rtc': by_mode['rtc'], 'guided': by_mode['guided']}
# excluded: nv[0:40] = 30k-checkpoint arms, nv[60:80] = objyaw (yaw pinned)

rows = []
for bname, rs in blocks.items():
    for r in rs:
        t = np.array(r['traj'])
        o = np.array(r['obj_xy'])
        d = np.linalg.norm(t[:, 0:2] - o, axis=1)
        k = int(d.argmin())
        dev = (o - mu)
        rows.append(dict(block=bname, miss=d[k],
                         off=t[k, 0:2] - o,
                         maha=float(np.sqrt(((dev / sd) ** 2).sum())),
                         devx=dev[0], devy=dev[1]))
print('pooled n =', len(rows))

miss = np.array([r['miss'] for r in rows])
maha = np.array([r['maha'] for r in rows])
offx = np.array([r['off'][0] for r in rows])
offy = np.array([r['off'][1] for r in rows])
devx = np.array([r['devx'] for r in rows])
devy = np.array([r['devy'] for r in rows])
blk = np.array([r['block'] for r in rows])

# mode as covariate: demean miss within block
missd = miss.copy()
for b in np.unique(blk):
    missd[blk == b] -= miss[blk == b].mean()


def corr(a, b):
    r = np.corrcoef(a, b)[0, 1]
    rs = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    return r, rs


r1, s1 = corr(maha, miss)
r2, s2 = corr(maha, missd)
print()
print('RADIAL: miss vs Mahalanobis spawn distance')
print('  raw     pearson %+.3f  spearman %+.3f' % (r1, s1))
print('  demeaned pearson %+.3f  spearman %+.3f   (n=%d, |r|>0.24 ~ p<0.05)'
      % (r2, s2, len(rows)))
lo = miss[maha <= np.median(maha)]
hi = miss[maha > np.median(maha)]
print('  near-prior half med %.3f | far half med %.3f'
      % (np.median(lo), np.median(hi)))

rx, sx = corr(devx, offx)
ry, sy = corr(devy, offy)
print()
print('DIRECTIONAL: miss component vs object deviation, same world axis')
print('  x (wide axis, sd %.2f): pearson %+.3f spearman %+.3f  slope %.3f'
      % (sd[0], rx, sx, np.polyfit(devx, offx, 1)[0]))
print('  y (narrow axis, sd %.2f): pearson %+.3f spearman %+.3f  slope %.3f'
      % (sd[1], ry, sy, np.polyfit(devy, offy, 1)[0]))
print()
print('  reading: slope ~ -1 on an axis = policy IGNORES that axis of the')
print('  object position and flies to the prior (miss grows 1:1 opposite the')
print('  deviation). slope ~ 0 = policy tracks the object on that axis.')
print()
for b in np.unique(blk):
    m = blk == b
    rb, sb = corr(maha[m], miss[m])
    print('  block %-6s n=%2d  miss med %.3f  pearson %+.3f' %
          (b, m.sum(), np.median(miss[m]), rb))
