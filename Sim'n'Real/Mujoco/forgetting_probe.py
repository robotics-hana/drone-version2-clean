"""forgetting_probe.py -- has fine-tuning destroyed pi0's pretrained
open-vocabulary grounding?

The mustard bottle -- a canonical YCB object that pi0's pretraining knows,
retired from this project's task set before dataset v2 -- is placed at the
trained task position with a TRAINED object (the weight) as distractor at
the trained distractor position. Two arms on identical scenes:

  bottle : "pick up the mustard bottle and put it in the wooden box"
  control: "pick up the weight and put it in the wooden box"

If the fine-tune preserved grounding, the closest approach follows the
prompt. If it caused catastrophic forgetting of object vocabulary, both
arms fly to the weight (or to the spawn prior) regardless of prompt.

Primary metric: closest-approach distance to the PROMPTED object.
Secondary: selection rate (which object the jaws end nearer). Success is
not scored -- the platform's grasp weld is bound to trained objects and a
bottle grasp is not the question.

Uses eval_pi0's repaired platform and drivers unmodified (no file edited);
scenes are the trained geometry with the bottle substituted post-reset.
Torch-seeded; PROV line printed first.
"""
import hashlib
import json
import sys
import time

import numpy as np
import mujoco
import torch

import collect_airvla as A
import eval_pi0 as E
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT = sys.argv[1]
ARM = sys.argv[2]                        # bottle | control
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
TORCHSEED = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

# swap arms: positions exchanged so language and position DISAGREE --
# the first arm-pair showed selection follows the task position regardless
# of prompt (both arms 15/20 to the task spot), because training always
# placed the prompted object there. Only a position/language conflict can
# test whether the object vocabulary survived fine-tuning.
PROMPT = {"bottle": "pick up the mustard bottle and put it in the wooden box",
          "control": "pick up the weight and put it in the wooden box",
          "bottle_swap": "pick up the mustard bottle and put it in the wooden box",
          "control_swap": "pick up the weight and put it in the wooden box"}[ARM]
SWAP = ARM.endswith("_swap")

KEY = {"observation.images.camera3": "observation.images.base_0_rgb",
       "observation.images.camera1": "observation.images.left_wrist_0_rgb",
       "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

policy = PI0Policy.from_pretrained(CKPT)
torch.manual_seed(TORCHSEED)
torch.cuda.manual_seed_all(TORCHSEED)
E.PRE, E.POST = make_pre_post_processors(policy.config, pretrained_path=CKPT)
policy = policy.to("cuda")
policy.eval()
E.PLATFIX[0] = 1                          # repaired platform automaton

print("PROV " + json.dumps(dict(
    script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
    ckpt=CKPT, arm=ARM, prompt=PROMPT, n=N, torch_seed=TORCHSEED,
    runner_seed=88000, when=time.strftime("%Y-%m-%dT%H:%M:%S"))), flush=True)

r = A.Runner(88000)                       # fresh seed family: scenes pair
driver = E.NaiveDriver(policy, torch, "cuda", torch.float32, KEY)
m = r.model
badr = r.badr                             # mustard bottle free joint
wadr = r.objs["weight"]["adr"]
padr = r.objs["plush penguin"]["adr"]
rng = np.random.default_rng(88000)        # scene sampling: same in both arms

for ep in range(N):
    # trained pick geometry: task spot on the table front edge, deck start
    bxy = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.20, 1.00)])
    for _ in range(300):
        start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.6),
                          rng.uniform(0.21, 0.30)])
        if (np.linalg.norm(start[0:2] - bxy) >= 0.70
                and start[1] >= bxy[1] + 0.35):
            break
    r.reset_scene(bxy, start, obj="weight")
    # substitution: penguin off-stage; weight -> the distractor spot it
    # vacated; BOTTLE upright at the task spot
    dist_xy = r.data.qpos[padr:padr + 2].copy()
    r.data.qpos[padr:padr + 2] = (-0.95, 2.60)
    if SWAP:
        # weight keeps the task spot; bottle takes the distractor spot
        r.data.qpos[badr:badr + 2] = dist_xy
    else:
        r.data.qpos[wadr:wadr + 2] = dist_xy
        r.data.qpos[badr:badr + 2] = bxy
    r.data.qpos[badr + 2] = A.PLATE_TOP + 0.10
    r.data.qpos[badr + 3:badr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, r.data)
    for _ in range(200):                  # settle the bottle onto the table
        mujoco.mj_step(m, r.data)
    plat = E.Platform(r, start, 0.0)
    plat.kind = "pick"
    driver.reset()
    d_b, d_w, zs = [], [], []
    for t in range(1200):
        act = driver.act(r, PROMPT)
        plat.tick(act)
        if t % 3 == 0:
            jaws = r.jaws()
            d_b.append(float(np.linalg.norm(jaws[0:2]
                                            - r.data.qpos[badr:badr + 2])))
            d_w.append(float(np.linalg.norm(jaws[0:2]
                                            - r.data.qpos[wadr:wadr + 2])))
            zs.append(float(r.data.qpos[2]))
    d_b, d_w = np.array(d_b), np.array(d_w)
    prompted_is_bottle = ARM.startswith("bottle")
    sel = "bottle" if d_b.min() < d_w.min() else "weight"
    print("EP " + json.dumps(dict(
        ep=ep, arm=ARM, bottle_xy=[round(float(x), 3) for x in bxy],
        min_d_bottle_mm=round(d_b.min() * 1000, 1),
        min_d_weight_mm=round(d_w.min() * 1000, 1),
        d_prompted_mm=round((d_b if prompted_is_bottle else d_w).min() * 1000, 1),
        selected=sel, zrange=round(max(zs) - min(zs), 2))), flush=True)

print("FORGETTING-PROBE-DONE arm=%s" % ARM, flush=True)
