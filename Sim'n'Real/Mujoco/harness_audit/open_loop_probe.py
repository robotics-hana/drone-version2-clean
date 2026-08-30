"""open_loop_probe.py CKPT [n] -- did the model learn the mapping, or not?

Teacher-forced: feed TRAINING frames, compare the predicted action chunk to the
logged one. No sim, no rollout, so closed-loop compounding cannot contaminate
the result. This separates the two hypotheses that the rollout evidence cannot:

  predictions track ground truth   -> mapping learned; failure is closed-loop
  predictions near-constant        -> collapse to the marginal; everything
                                      downstream (delta parameterisation,
                                      cameras, chunk horizon) is a spectator

The last two lines are the experiment: pred spread vs gt spread ACROSS frames.
"""
import sys, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
H, FPS = 50, 10
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
KEY = {'observation.images.camera3': 'observation.images.base_0_rgb',
       'observation.images.camera1': 'observation.images.left_wrist_0_rgb',
       'observation.images.camera2': 'observation.images.right_wrist_0_rgb'}

ds = LeRobotDataset('hanapasta/airvla_full',
                    delta_timestamps={'action': [i / FPS for i in range(H)]})
pol = PI0Policy.from_pretrained(CKPT).to(DEV).eval()
PRE, POST = make_pre_post_processors(pol.config, pretrained_path=CKPT)
print('device', DEV, '| dataset frames', len(ds), '| samples', N, flush=True)

idx = np.random.default_rng(0).choice(len(ds), N, replace=False)
P, G = [], []
for j, i in enumerate(idx):
    s = ds[int(i)]
    b = {v: s[k].unsqueeze(0).to(DEV) for k, v in KEY.items()}
    b['observation.state'] = s['observation.state'].unsqueeze(0).to(DEV)
    b['task'] = s['task']
    with torch.no_grad():
        a = POST(pol.predict_action_chunk(PRE(b)))
    a = a['action'] if isinstance(a, dict) else a
    P.append(np.asarray(a.float().cpu())[0]); G.append(np.asarray(s['action']))
    if (j + 1) % 10 == 0: print(' %d/%d' % (j + 1, N), flush=True)
P, G = np.array(P), np.array(G)

names = ['dx','dy','dz','droll','dpitch','dyaw','grip']
print()
print('dim         MAE      corr    pred_sd     gt_sd')
for d, n in enumerate(names):
    p, g = P[:, :, d].ravel(), G[:, :, d].ravel()
    c = np.corrcoef(p, g)[0, 1] if p.std() > 1e-9 and g.std() > 1e-9 else float('nan')
    print('%-7s %9.5f  %+.3f  %9.5f %9.5f' % (n, np.abs(p - g).mean(), c, p.std(), g.std()))
ep, eg = P[:, :, 0:3].sum(1), G[:, :, 0:3].sum(1)
print()
print('5s integrated endpoint err (m): %.4f  | gt endpoint magnitude (m): %.4f'
      % (float(np.linalg.norm(ep - eg, axis=1).mean()), float(np.linalg.norm(eg, axis=1).mean())))
print('pred spread across frames:', np.round(P.mean(1).std(0), 5))
print('gt   spread across frames:', np.round(G.mean(1).std(0), 5))
r = P.mean(1).std(0) / np.maximum(G.mean(1).std(0), 1e-9)
print('ratio pred/gt            :', np.round(r, 3))
