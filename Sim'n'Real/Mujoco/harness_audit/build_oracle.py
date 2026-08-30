"""build_oracle.py -- derive the oracle jaws-to-target vector for all 400
episodes and write dataset `airvla_oracle` with observation.state extended
from 10 to 13 dims.

Diagnostic rationale (reviewer, 2026-08-30): this bounds the problem from
above. The vector is privileged and would never exist on a deployed system,
but if a policy HANDED the exact gripper-to-target vector grasps reliably,
everything downstream of perception works and the remaining problem is
seeing; if it still misses laterally, perception was never the constraint
and the fault is control/action-representation.

Cheap by construction: pure forward kinematics from logged state (drone
free-joint pose + arm joints + aperture) and scene_state (object pose) --
no rendering, no simulation stepping, no recollection. pi0 pads state to
32 dims, so 10->13 needs no architecture change.

Videos are HARDLINKED (same filesystem), parquet rewritten, meta/stats
state entry recomputed for 13 dims.
"""
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
import mujoco
import collect_airvla as A

SRC = os.path.expanduser('~/Scratch/hf_cache/lerobot/hanapasta/airvla_full')
DST = os.path.expanduser('~/Scratch/hf_cache/lerobot/hanapasta/airvla_oracle')

r = A.Runner(50000)
m, d = r.model, r.data
j1 = m.joint('Joint_1').qposadr[0]
j2 = m.joint('Joint_2').qposadr[0]
gadr = m.joint('right_clamp').qposadr[0]
AIMZ = {k: v['aim_z'] for k, v in r.objs.items()}
print('aim_z:', AIMZ, flush=True)

ep = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob(SRC + '/meta/episodes/chunk-000/*.parquet'))])
ep = ep.sort_values('episode_index')

# ---- copy tree: videos hardlinked, meta copied, data rewritten ----
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)
shutil.copytree(SRC + '/meta', DST + '/meta')
for root, dirs, files in os.walk(SRC + '/videos'):
    rel = os.path.relpath(root, SRC)
    os.makedirs(os.path.join(DST, rel), exist_ok=True)
    for f in files:
        os.link(os.path.join(root, f), os.path.join(DST, rel, f))
print('videos hardlinked', flush=True)

os.makedirs(DST + '/data/chunk-000', exist_ok=True)
ORA = []
for pf in sorted(glob.glob(SRC + '/data/chunk-000/*.parquet')):
    df = pd.read_parquet(pf)
    S = np.stack(df['observation.state'].to_numpy())
    SC = np.stack(df['scene_state'].to_numpy())
    ei = df['episode_index'].to_numpy()
    tasks = {int(row['episode_index']): str(row['tasks'])
             for _, row in ep.iterrows()}
    out = np.zeros((len(df), 3), np.float32)
    for i in range(len(df)):
        d.qpos[0:7] = S[i, 0:7]
        d.qpos[gadr] = S[i, 7] * 0.016
        d.qpos[j1] = S[i, 8]
        d.qpos[j2] = S[i, 9]
        mujoco.mj_kinematics(m, d)
        jaws = np.array(A.C.grasp_site_pos(m, d))
        obj = 'plush penguin' if 'penguin' in tasks[int(ei[i])] else 'weight'
        aim = SC[i, 0:3].astype(float).copy()
        aim[2] += AIMZ[obj]
        out[i] = (aim - jaws).astype(np.float32)
    ORA.append(out)
    new = np.concatenate([S.astype(np.float32), out], axis=1)
    df['observation.state'] = list(new)
    df.to_parquet(DST + '/data/chunk-000/' + os.path.basename(pf))
    print('wrote', os.path.basename(pf), len(df), flush=True)

ORA = np.concatenate(ORA)
print('oracle vector: mean', ORA.mean(0).round(3), 'std', ORA.std(0).round(3),
      flush=True)

# ---- meta: state feature 10 -> 13, stats extended ----
info = json.load(open(DST + '/meta/info.json'))
f = info['features']['observation.state']
f['shape'] = [13]
f['names'] = list(f['names']) + ['orax', 'oray', 'oraz'] \
    if isinstance(f.get('names'), list) else f.get('names')
json.dump(info, open(DST + '/meta/info.json', 'w'), indent=4)

stats = json.load(open(DST + '/meta/stats.json'))
st = stats['observation.state']
q = {'min': ORA.min(0), 'max': ORA.max(0), 'mean': ORA.mean(0),
     'std': ORA.std(0),
     'q01': np.quantile(ORA, .01, 0), 'q10': np.quantile(ORA, .10, 0),
     'q50': np.quantile(ORA, .50, 0), 'q90': np.quantile(ORA, .90, 0),
     'q99': np.quantile(ORA, .99, 0)}
for k in list(st.keys()):
    if k == 'count':
        continue
    if k in q:
        st[k] = list(np.asarray(st[k], float).ravel()) + \
            [float(x) for x in q[k]]
json.dump(stats, open(DST + '/meta/stats.json', 'w'), indent=4)
print('meta updated', flush=True)

# ---- per-episode stats parquet also carries observation.state stats ----
for pf in sorted(glob.glob(DST + '/meta/episodes/chunk-000/*.parquet')):
    df = pd.read_parquet(pf)
    cols = [c for c in df.columns if c.startswith('stats/observation.state/')]
    for c in cols:
        base = c.split('/')[-1]
        if base == 'count':
            continue
        vals = df[c].to_numpy()
        fixed = []
        for v in vals:
            arr = np.asarray(v, dtype=np.float32).ravel()
            if arr.shape[0] == 10:
                add = q.get(base)
                arr = np.concatenate([arr, np.asarray(
                    add if add is not None else [0, 0, 0], np.float32).ravel()])
            fixed.append(arr)
        df[c] = fixed
    df.to_parquet(pf)
print('episode stats extended', flush=True)

# ---- smoke test: load through LeRobot ----
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('hanapasta/airvla_oracle')
s = ds[100]
print('LOAD OK  state shape:', tuple(s['observation.state'].shape),
      'sample tail:', s['observation.state'][-3:].numpy().round(3), flush=True)
print('ORACLE-BUILD-DONE', flush=True)
