"""audit_analysis.py -- the three corrected measurements plus the free test.

Fixes applied relative to the discarded 26-episode analysis:
  1. episodes selected by the run's `tag` field, not by counting backwards;
  2. distances measured from the JAWS (logged per sample), not the airframe;
  3. lateral bias computed in the BODY frame via logged yaw -- the quantity
     Hana actually observed in the wrist video -- not relative to a fixed
     world line the drone never sees.
Plus: miss vs distance-of-object-from-training-mean (spawn correlation).
  Strong positive correlation -> the policy flies to a prior it cannot
  correct visually -> the dead base camera is the binding constraint.
  Flat -> the policy sees but is imprecise -> training/horizon story.
"""
import json
import numpy as np

TAG = 'audit60k'
F = '/home/ucabhe0/Scratch/airvla/evalout/eval_traj.jsonl'
recs = [json.loads(l) for l in open(F)]
seg = [r for r in recs if r.get('tag') == TAG and r['kind'] == 'pick']
print('episodes with tag=%s: %d' % (TAG, len(seg)))
assert seg, 'no tagged episodes'

train = np.load('/home/ucabhe0/Scratch/airvla/evalout/train_obj_xy.npy')
mu = train.mean(0)
print('training object mean (%.3f, %.3f)' % (mu[0], mu[1]))

rows = []
for r in seg:
    t = np.array(r['traj'])            # cols: x y z yaw jx jy jz
    o = np.array(r['obj_xy'])
    dj = np.linalg.norm(t[:, 4:6] - o, axis=1)      # JAWS xy distance
    k = int(dj.argmin())
    off_w = t[k, 4:6] - o                           # world-frame jaw offset
    yaw = t[k, 3]
    # body frame: x_b = right(?), derived from yaw rotation of world offset.
    # +lat = object appears to the drone's LEFT <=> jaws sit RIGHT of object;
    # sign convention printed either way -- consistency is what matters.
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    off_b = np.array([c * off_w[0] - s_ * off_w[1],
                      s_ * off_w[0] + c * off_w[1]])
    db = np.linalg.norm(t[:, 0:2] - o, axis=1)      # body xy distance
    rows.append(dict(miss=dj[k], off_w=off_w, off_b=off_b,
                     body_min=db.min(), spawn_d=np.linalg.norm(o - mu),
                     zr=t[:, 2].max() - t[:, 2].min()))

miss = np.array([r['miss'] for r in rows])
ob = np.array([r['off_b'] for r in rows])
ow = np.array([r['off_w'] for r in rows])
bm = np.array([r['body_min'] for r in rows])
sd = np.array([r['spawn_d'] for r in rows])
zr = np.array([r['zr'] for r in rows])

print()
print('1. JAWS closest xy miss: med %.3f m  mean %.3f  min %.3f  max %.3f'
      % (np.median(miss), miss.mean(), miss.min(), miss.max()))
print('   body closest xy dist: med %.3f m  (jaw-vs-body gap: %.3f)'
      % (np.median(bm), np.median(bm) - np.median(miss)))
print('   z-range med %.2f m (takeoff flown in %d/%d)'
      % (np.median(zr), int((zr > 0.2).sum()), len(zr)))
print()
print('2. BODY-frame jaw offset at closest approach')
print('   AXES: nose = body -y. x_b = LATERAL (the axis Hana saw);'
      ' y_b = fore-aft, NEGATIVE = ahead of the aircraft.')
print('   x_b: mean %+.3f sd %.3f  |  y_b: mean %+.3f sd %.3f'
      % (ob[:, 0].mean(), ob[:, 0].std(), ob[:, 1].mean(), ob[:, 1].std()))
print('   consistent-side fraction: x_b %.2f  y_b %.2f'
      % (max((ob[:, 0] > 0).mean(), (ob[:, 0] < 0).mean()),
         max((ob[:, 1] > 0).mean(), (ob[:, 1] < 0).mean())))
print('   (world-frame, for reference: dx %+.3f+-%.3f  dy %+.3f+-%.3f)'
      % (ow[:, 0].mean(), ow[:, 0].std(), ow[:, 1].mean(), ow[:, 1].std()))
print()
r_p = np.corrcoef(sd, miss)[0, 1]
rk_s = np.argsort(np.argsort(sd))
rk_m = np.argsort(np.argsort(miss))
r_s = np.corrcoef(rk_s, rk_m)[0, 1]
print('3. SPAWN CORRELATION  miss vs |obj - train_mean|:')
print('   pearson r = %+.3f   spearman rho = %+.3f   n = %d'
      % (r_p, r_s, len(miss)))
print('   spawn_d: med %.3f  range [%.3f, %.3f]'
      % (np.median(sd), sd.min(), sd.max()))
lo = miss[sd <= np.median(sd)]
hi = miss[sd > np.median(sd)]
print('   near-prior half: miss med %.3f | far half: miss med %.3f'
      % (np.median(lo), np.median(hi)))
print()
print('per-episode (spawn_d, miss, off_b):')
for r in sorted(rows, key=lambda r: r['spawn_d']):
    print('   %.3f  %.3f  (%+.3f, %+.3f)'
          % (r['spawn_d'], r['miss'], r['off_b'][0], r['off_b'][1]))
