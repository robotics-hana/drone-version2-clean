"""collect_dagger.py -- FT-DAG on-policy corrective collection
(pre-registered 2026-09-14, Hana: "let's try this").

Roll-in policy, roll-out expert: Plan C (C-15000) drives the
platform on a fresh collection scene with NOTHING recorded; when it
parks near the commanded object (jaws within NEAR_R for NEAR_HOLD
ticks) or the roll-in budget lapses, the expert's hidden
integrators are synced to the platform state and the STOCK
pick_v2 -> place_v2 pipeline completes the episode. Recorded frames
start AT the seam, so the banked data is (policy-visited state,
expert-consistent completion) -- chunk-consistent DAgger for a
chunked policy. Policy actions are never supervision; roll-in
contact counters are zeroed at takeover (roll-in hits go to the
manifest only); a roll-in that welds by itself is discarded.

usage (GPU node):
  python collect_dagger.py <seed> <n_units> [hours] [repo_id]
defaults repo hanapasta/airvla_dag; manifest dag_manifest_<seed>.jsonl
"""
import json
import sys
import time

import numpy as np
import torch

import collect_airvla as A
import collect_v2 as V
from platform_v2 import V2Platform
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

CKPT = ("/home/ucabhe0/Scratch/airvla/pi0_e3c_out/checkpoints/"
        "015000/pretrained_model")
SEED = int(sys.argv[1])
N_UNITS = int(sys.argv[2])
HOURS = float(sys.argv[3]) if len(sys.argv) > 3 else None
REPO = sys.argv[4] if len(sys.argv) > 4 else "hanapasta/airvla_dag"

HORIZON = 50
ROLL_TICKS = 600                 # roll-in cap (policy parks by ~400)
NEAR_R, NEAR_HOLD = 0.25, 25     # sustained-near takeover trigger

KEY = {"observation.images.camera3": "observation.images.base_0_rgb",
       "observation.images.camera1": "observation.images.left_wrist_0_rgb",
       "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

policy = PI0Policy.from_pretrained(CKPT)
torch.manual_seed(1000)
torch.cuda.manual_seed_all(1000)
PRE, POST = make_pre_post_processors(policy.config, pretrained_path=CKPT)
policy = policy.to("cuda").eval()
print("DAG policy loaded:", CKPT, flush=True)

dataset = V._make_v2_dataset(REPO)
r = V.V2Runner(SEED)
man = open("dag_manifest_%d.jsonl" % SEED, "a")


def obs_batch(task):
    f = r.frame(task)
    batch = {}
    for ck, slot in KEY.items():
        t = torch.from_numpy(f[ck]).permute(2, 0, 1).float() / 255.0
        batch[slot] = t.unsqueeze(0).cuda()
    batch["observation.state"] = torch.from_numpy(
        np.asarray(f["observation.state"],
                   dtype=np.float32)).unsqueeze(0).cuda()
    batch["task"] = [task]
    return batch


def dagger_episode():
    """One unit: policy roll-in (unrecorded) -> expert completion
    (recorded). Returns (frames, res) like episode_v2, with seam
    metadata merged into res; frames is None on discard."""
    obj, start, alt, tgt = r.reset_scene_v2()
    task = A.PROMPT_MANIP.format(obj=obj)
    plat = V2Platform(r, start, 0.0)
    near = 0
    ticks_in = 0
    stop = False
    for _ in range(ROLL_TICKS // HORIZON):
        with torch.no_grad():
            chunk = POST(policy.predict_action_chunk(
                PRE(obs_batch(task))))[0].float().cpu().numpy()[:HORIZON]
        for a in chunk:
            plat.tick(a)
            ticks_in += 1
            if bool(r.data.eq_active[r.weld]):
                return None, dict(obj=obj, reason="policy-welded-rollin",
                                  rollin_ticks=ticks_in)
            aim_l, _ = r.live_target()
            dxy = float(np.linalg.norm(r.jaws()[0:2] - aim_l[0:2]))
            near = near + 1 if dxy < NEAR_R else 0
            if near >= NEAR_HOLD:
                stop = True
                break
        if stop:
            break
    aim_l, _ = r.live_target()
    seam = dict(rollin_ticks=ticks_in,
                trigger=("near" if stop else "timeout"),
                seam_jaw_dxy_mm=round(1000 * float(np.linalg.norm(
                    r.jaws()[0:2] - aim_l[0:2])), 1),
                rollin_table_hits=r._table_hits,
                rollin_obj_hits=r._obj_hits)
    # EXPERT TAKEOVER: sync the expert's hidden integrators to the
    # platform's (sp is a hidden integrator on BOTH sides -- the
    # training-deployment-parity lesson), reopen the grip, and hand
    # the episode to the stock pipeline. Recording starts here.
    ex = r.expert
    ex.sp = np.array(plat.sp, dtype=float)
    ex.goal = ex.sp.copy()
    ex.yaw = float(plat.yaw)
    ex.goal_yaw = float(plat.yaw)
    ex.grip = float(plat.grip)
    ex.goal_grip = 1.0
    r._table_hits = 0
    r._obj_hits = 0
    frames = []
    grasped = r.pick_v2(frames, task, alt)
    d_bin = r.place_v2(frames, task) if grasped else float("nan")
    res = dict(obj=obj, corrective=False, terminal=False,
               grasped=grasped,
               d_bin_mm=(round(1000 * d_bin, 1) if grasped else None),
               ticks=len(frames), table_hits=r._table_hits,
               obj_hits=r._obj_hits,
               clean=(r._table_hits == 0 and r._obj_hits == 0))
    res.update(seam)
    return frames, res


def main():
    saved = attempts = 0
    t0 = time.time()

    def out_of_time():
        return HOURS is not None and (time.time() - t0) / 3600 > HOURS

    for slot in range(N_UNITS):
        if out_of_time():
            print("TIME-BUDGET STOP before slot %d" % slot, flush=True)
            break
        for _ in range(6):
            attempts += 1
            rec = dict(attempt=attempts, slot=slot, kind="dagger",
                       collector_seed=SEED)
            try:
                frames, res = dagger_episode()
            except Exception as e:      # noqa: BLE001 -- log + retry
                rec.update(banked=False, reason="exception:%s" % e)
                man.write(json.dumps(rec) + "\n")
                man.flush()
                continue
            rec.update(res)
            if frames is None:
                rec.update(banked=False)
                man.write(json.dumps(rec) + "\n")
                man.flush()
                continue
            ok, reason = V._judge_pick(res)
            rec.update(banked=ok, reason=reason,
                       parking=V.parking_window(frames))
            man.write(json.dumps(rec) + "\n")
            man.flush()
            if not ok:
                print("attempt %d REJECTED: %s" % (attempts, reason),
                      flush=True)
                continue
            if not V._bank_episode(dataset, frames):
                rec2 = dict(rec, banked=False, reason="writer-error")
                man.write(json.dumps(rec2) + "\n")
                man.flush()
                continue
            saved += 1
            print("BANKED %d/%d (slot %d, %s seam=%.0fmm rollin=%d, "
                  "%.1f h)" % (saved, N_UNITS, slot, res["trigger"],
                               res["seam_jaw_dxy_mm"],
                               res["rollin_ticks"],
                               (time.time() - t0) / 3600), flush=True)
            break
    print("COLLECT-DAG-DONE %d/%d banked in %d attempts"
          % (saved, N_UNITS, attempts), flush=True)


main()
