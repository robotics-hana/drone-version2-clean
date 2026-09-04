"""eval_v2.py -- closed-loop policy evaluation in the V2 WORLD.

The v2 policy was trained on the v2 scene (fixed table, blue penguin,
box beside the table, 35% legs, camera3 at the measured workspace pose).
The frozen v1 harness renders the v1 world -- old camera3, sliding
table, far bin -- so running v2 through it would measure distribution
shift, not the policy. This harness keeps the v1 protocol's DISCIPLINE
(metrics, seeding, PROV, trajectory logging, no-flag-drift) on the v2
world. v1-vs-v2 stays two studies, as pre-registered.

Scene + cameras come from collect_v2.V2Runner (the exact collection
code); the policy flies closed-loop from its three camera streams +
state, 50-step chunks executed open-loop then re-planned (naive mode,
matching the v1 protocol's naive condition).

Metrics per pick episode: miss_mm (closest jaw-to-target, primary),
picked (weld-quality equivalent: object lifted >12 cm while within the
jaw window), placed/success (object inside the box, released), plus the
v2 gates as OBSERVATIONS (table/object/gate contacts are reported, not
enforced -- the policy is being measured, not curated).

usage:
  python eval_v2.py <ckpt> <n_pick> <n_nav> --torchseed 1000 --tag TAG
"""
import hashlib
import json
import sys
import time

import numpy as np
import mujoco
import torch

import collect_airvla as A
import collect_v2 as V
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT = sys.argv[1]
N_PICK = int(sys.argv[2])
N_NAV = int(sys.argv[3])
TORCHSEED = int(sys.argv[4][12:]) if "--torchseed=" in " ".join(sys.argv) \
    else (int(sys.argv[sys.argv.index("--torchseed") + 1])
          if "--torchseed" in sys.argv else 1000)
TAG = (sys.argv[sys.argv.index("--tag") + 1]
       if "--tag" in sys.argv else "v2eval")
SCENE_SEED = 97000                     # NEW family: disjoint from
                                       # collection (71000), v1 eval
                                       # (77000), probes (88000)
HORIZON = 50
KEY = {"observation.images.camera3": "observation.images.base_0_rgb",
       "observation.images.camera1": "observation.images.left_wrist_0_rgb",
       "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

policy = PI0Policy.from_pretrained(CKPT)
torch.manual_seed(TORCHSEED)
torch.cuda.manual_seed_all(TORCHSEED)
PRE, POST = make_pre_post_processors(policy.config, pretrained_path=CKPT)
policy = policy.to("cuda").eval()

print("PROV " + json.dumps(dict(
    script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
    ckpt=CKPT, argv=sys.argv[1:], torch_seed=TORCHSEED,
    scene_seed=SCENE_SEED, tag=TAG,
    when=time.strftime("%Y-%m-%dT%H:%M:%S"))), flush=True)

r = V.V2Runner(SCENE_SEED)
IMG224 = 224


def obs_batch(task):
    f = r.frame(task)                  # v2 cameras incl. cam3_v2
    batch = {}
    for cam_key, slot in KEY.items():
        img = f[cam_key]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        batch[slot] = t.unsqueeze(0).cuda()
    st = torch.from_numpy(np.asarray(f["observation.state"],
                                     dtype=np.float32))
    batch["observation.state"] = st.unsqueeze(0).cuda()
    batch["task"] = [task]
    return batch


def run_pick(i):
    obj, start, alt, tgt = r.reset_scene_v2()
    task = A.PROMPT_MANIP.format(obj=obj)
    ex = r.expert
    adr = r.cur["adr"]
    miss = 1e9
    lifted = False
    frames_n = 0
    traj = []
    for chunk_i in range(24):          # 24 x 50 ticks = 120 s cap
        with torch.no_grad():
            batch = PRE(obs_batch(task))
            chunk = policy.predict_action_chunk(batch)
            chunk = POST(chunk)[0].float().cpu().numpy()[:HORIZON]
        for a in chunk:
            ex.sp = ex.sp + np.clip(a[0:3], -0.035, 0.035)
            ex.yaw = ex.yaw + float(np.clip(a[5], -0.2, 0.2))
            ex.grip = float(np.clip(a[6], 0.0, 1.0))
            r.ctrl.set_targets(ex.sp, ex.arm,
                               A.C.GRIPPER_OPEN * ex.grip)
            r.ctrl.mppi.target_yaw = ex.yaw
            for _ in range(r.sub):
                r.ctrl.step()
                mujoco.mj_step(r.model, r.data)
            frames_n += 1
            aim, _ = r.live_target()
            d = float(np.linalg.norm(r.jaws() - aim))
            miss = min(miss, d)
            oz = float(r.data.qpos[adr + 2])
            if oz > A.MAT_TOP + 0.12 and d < 0.06:
                lifted = True
            if frames_n % 3 == 0:
                traj.append([round(float(x), 3) for x in
                             (*r.data.qpos[0:3], ex.yaw, *r.jaws())])
    oxy = r.data.qpos[adr:adr + 2]
    d_bin = float(np.linalg.norm(oxy - r.bin_xy))
    placed = bool(d_bin <= 0.15
                  and float(r.data.qpos[adr + 2]) < A.MAT_TOP + 0.25)
    res = dict(kind="pick", ep=i, obj=obj, tag=TAG,
               miss_mm=round(miss * 1000, 1), picked=bool(lifted),
               placed=placed, success=placed,
               d_bin_mm=round(d_bin * 1000, 1),
               table_hits=r._table_hits, obj_hits=r._obj_hits,
               frames=frames_n)
    with open("eval_v2_traj.jsonl", "a") as fh:
        fh.write(json.dumps(dict(tag=TAG, kind="pick", ep=i, obj=obj,
                                 tgt=[round(float(x), 3) for x in tgt],
                                 bin_xy=[round(float(x), 3)
                                         for x in r.bin_xy],
                                 traj=traj)) + "\n")
    return res


def run_nav(i):
    obj, start, alt, tgt = r.reset_scene_v2(nav=True)
    task = A.PROMPT_NAV.format(obj=obj)
    ex = r.expert
    adr = r.cur["adr"]
    gx = r._gate_x
    crossed = False
    hover_run = 0
    hover_ok = False
    y_prev = float(r.data.qpos[1])
    frames_n = 0
    for chunk_i in range(10):          # 50 s cap
        with torch.no_grad():
            batch = PRE(obs_batch(task))
            chunk = policy.predict_action_chunk(batch)
            chunk = POST(chunk)[0].float().cpu().numpy()[:HORIZON]
        for a in chunk:
            ex.sp = ex.sp + np.clip(a[0:3], -0.035, 0.035)
            ex.yaw = ex.yaw + float(np.clip(a[5], -0.2, 0.2))
            ex.grip = float(np.clip(a[6], 0.0, 1.0))
            r.ctrl.set_targets(ex.sp, ex.arm,
                               A.C.GRIPPER_OPEN * ex.grip)
            r.ctrl.mppi.target_yaw = ex.yaw
            for _ in range(r.sub):
                r.ctrl.step()
                mujoco.mj_step(r.model, r.data)
            frames_n += 1
            y = float(r.data.qpos[1])
            if (y_prev < -0.6 <= y
                    and abs(float(r.data.qpos[0]) - gx) < 0.45
                    and 0.38 < float(r.data.qpos[2]) < 1.44):
                crossed = True
            y_prev = y
            near = float(np.linalg.norm(
                r.data.qpos[0:2] - r.data.qpos[adr:adr + 2])) < 0.25
            hover_run = hover_run + 1 if (crossed and near) else 0
            if hover_run >= 30:
                hover_ok = True
    return dict(kind="nav", ep=i, obj=obj, tag=TAG,
                crossed=bool(crossed), hover=bool(hover_ok),
                success=bool(crossed and hover_ok),
                gate_hits=r._gate_hits, table_hits=r._table_hits,
                frames=frames_n)


results = []
for i in range(N_PICK):
    ex = r.expert
    res = run_pick(i)
    results.append(res)
    print("EVAL " + json.dumps(res), flush=True)
for i in range(N_NAV):
    res = run_nav(i)
    results.append(res)
    print("EVAL " + json.dumps(res), flush=True)

picks = [x for x in results if x["kind"] == "pick"]
if picks:
    mm = sorted(x["miss_mm"] for x in picks)
    print("PICK SUMMARY: n=%d median %.1f mm  picked %d  placed %d"
          % (len(picks), mm[len(mm) // 2],
             sum(x["picked"] for x in picks),
             sum(x["placed"] for x in picks)), flush=True)
navs = [x for x in results if x["kind"] == "nav"]
if navs:
    print("NAV SUMMARY: n=%d success %d/%d"
          % (len(navs), sum(x["success"] for x in navs), len(navs)),
          flush=True)
print("EVALV2-DONE", flush=True)
