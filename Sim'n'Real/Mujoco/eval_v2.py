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
  python eval_v2.py <ckpt> <n_pick> <n_nav> --torchseed 1000 --tag TAG \
      [--video]     # film each episode: v2vid_TAG_pick00.mp4 etc.
      [--exec N]    # execute N of 50 actions per chunk then replan
                    # (default 50 = the frozen naive baseline; E1=10)
"""
import hashlib
import json
import sys
import time

import numpy as np
import imageio.v2 as imageio
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
VIDEO = "--video" in sys.argv           # film episodes: cam1|cam2|cam3
                                        # strip, every 2nd tick, 10 fps
                                        # (identical to the demo videos)
SCENE_SEED = 97000                     # NEW family: disjoint from
                                       # collection (71000), v1 eval
                                       # (77000), probes (88000)
HORIZON = 50
# E1 (pre-registered 2026-09-05): actions EXECUTED per 50-step chunk
# before re-inferring. Default 50 = the frozen naive baseline -- with
# the flag absent the loop counts and executed actions are identical
# to every prior run. --exec 10 replans at 1 Hz instead of 0.2 Hz;
# the per-episode tick budgets are enforced exactly (divisibility
# asserted). NOTE: this block must sit BELOW the HORIZON definition
# (review 2026-09-05 caught a NameError from the original placement).
assert not any(a.startswith("--exec=") for a in sys.argv), \
    "use the space form '--exec N' ('--exec=N' would be silently ignored)"
EXEC = (int(sys.argv[sys.argv.index("--exec") + 1])
        if "--exec" in sys.argv else HORIZON)
assert 1 <= EXEC <= HORIZON, "exec horizon must be in [1, %d]" % HORIZON
PICK_TICKS = 1200                       # 24 x 50 in the baseline
NAV_TICKS = 500                         # 10 x 50 in the baseline
assert PICK_TICKS % EXEC == 0 and NAV_TICKS % EXEC == 0, \
    "exec horizon must divide both tick budgets (use 1/2/4/5/10/20/25/50)"
KEY = {"observation.images.camera3": "observation.images.base_0_rgb",
       "observation.images.camera1": "observation.images.left_wrist_0_rgb",
       "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

policy = PI0Policy.from_pretrained(CKPT)
torch.manual_seed(TORCHSEED)
torch.cuda.manual_seed_all(TORCHSEED)
PRE, POST = make_pre_post_processors(policy.config, pretrained_path=CKPT)
policy = policy.to("cuda").eval()

# hash the physics/scene dependencies too, not just this script --
# pairing across runs depends on them and cluster file drift is a
# documented hazard (review 2026-09-04)
DEPS = {}
for _m in ("collect_airvla", "collect_v2", "pd_flight", "collect_demos"):
    _mod = sys.modules.get(_m)
    if _mod is not None and getattr(_mod, "__file__", None):
        DEPS[_m] = hashlib.sha256(
            open(_mod.__file__, "rb").read()).hexdigest()[:12]
print("PROV " + json.dumps(dict(
    script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
    dep_sha=DEPS, exec_horizon=EXEC,
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


class V2Platform:
    """The arm/grasp automaton for v2 evaluation. Ported from the v1
    path, then REVALIDATED BY EXPERT-ACTION REPLAY in the v2 world
    (2026-09-04) which falsified two v1-heritage rules: the arm now
    deploys on sustained proximity to the target (the expert deploys
    after settling at the ~0.50 m standoff, never on the ascent), and
    the weld gate mirrors the collector's verbatim (10 mm/15 mm
    jaw-to-aim + per-object aperture window; NO pad-contact debounce
    -- the close-in-motion grasp welds before contact registers).
    Release when the policy opens the grip while welded (payload
    feed-forward handled inside V2Runner.weld_grasp)."""

    def __init__(self, rr, start, yaw0):
        self.r = rr
        self.sp = np.array(start, dtype=float)
        self.yaw = float(yaw0)
        self.grip = 1.0
        self.deployed = False
        self.near = 0
        self.cmd = np.array(rr.expert.q_travel, dtype=float)
        self.i = 0
        self.weld_tick = None
        self.released = False

    def tick(self, act, nav=False):
        rr = self.r
        self.sp = self.sp + np.clip(act[0:3], -0.035, 0.035)
        self.yaw += float(np.clip(act[5], -0.06, 0.06))
        self.grip = float(np.clip(act[6], 0.0, 1.0))
        welded = bool(rr.data.eq_active[rr.weld])
        ap = float(rr.data.qpos[rr.gadr])
        # weld gate (ground-truth-replay fix #2, 2026-09-04): mirror
        # the COLLECTOR's weld condition verbatim (collect_v2 pick_v2
        # creep_tick): jaw-to-aim within 10 mm horizontal / 15 mm
        # vertical AND aperture inside the PER-OBJECT window (mm).
        # The old v1-heritage gate (fixed 4-17 mm window + pad-contact
        # 2-of-4 debounce) is unsatisfiable in the v2 world: the
        # close-in-motion grasp welds on proximity before the pads
        # ever register contact, so expert-action replay could never
        # weld under it.
        if not nav and not welded:
            aim, _ = rr.live_target()
            pc = rr.jaws() - aim
            if (float(np.linalg.norm(pc[0:2])) < 0.010
                    and abs(float(pc[2])) < 0.015
                    and rr.cur["ap_lo"] < ap * 1000 < rr.cur["ap_hi"]):
                rr.weld_grasp(True)
        elif welded and self.grip > 0.8:
            rr.weld_grasp(False)
            self.released = True
        welded = bool(rr.data.eq_active[rr.weld])
        if welded and self.weld_tick is None:
            self.weld_tick = self.i
        # deploy rule (ground-truth-replay fix, 2026-09-04): the v2
        # expert deploys the arm only AFTER settling at the ~0.50 m
        # standoff and re-aiming (collect_v2 pick_v2) -- never on the
        # ascent. The old sp_z>=0.45-or-tick-60 rule (v1 heritage)
        # fired at tick ~5, destabilised the drone with an early arm
        # swing and left a ~200 mm offset through the grasp window;
        # expert-action replay could not weld. Proximity + debounce
        # mirrors the expert's observable cue.
        if not self.deployed and not nav:
            aim, _ = rr.live_target()
            dxy = float(np.linalg.norm(
                np.asarray(rr.data.qpos[0:2]) - np.asarray(aim[0:2])))
            self.near = self.near + 1 if dxy < 0.60 else 0
            if self.near >= 5:
                self.deployed = True
        tgt = (rr.expert.q_carry if (self.deployed and not nav
                                     and not self.released)
               else rr.expert.q_travel)
        self.cmd = self.cmd + np.clip(np.array(tgt) - self.cmd,
                                      -0.06, 0.06)
        self.i += 1
        rr.ctrl.set_targets(self.sp, self.cmd,
                            A.C.GRIPPER_OPEN * self.grip)
        rr.ctrl.mppi.target_yaw = self.yaw
        for _ in range(rr.sub):
            rr.ctrl.step()
            mujoco.mj_step(rr.model, rr.data)
        # contact observations. The collector tallies these inside
        # V2Runner.step, which eval never calls, so without this block
        # the reported table/object/gate hits are structurally 0
        # (dead-metric defect, review 2026-09-04). Same per-tick
        # cadence and dedupe as collect_v2.step; the collector's
        # jaws-outside-grasp-phase rule is omitted -- it keys on the
        # expert's phase flag, and the policy owns its grasp timing.
        d = rr.data
        table_hit = obj_hit = False
        for ci in range(d.ncon):
            g1 = int(d.contact.geom1[ci])
            g2 = int(d.contact.geom2[ci])
            if not table_hit and (
                    (g1 in rr._drone_geoms and g2 in rr._table_geoms)
                    or (g2 in rr._drone_geoms
                        and g1 in rr._table_geoms)):
                rr._table_hits += 1
                table_hit = True
            if not obj_hit and (
                    (g1 in rr._strike_geoms and g2 in rr._obj_geoms)
                    or (g2 in rr._strike_geoms
                        and g1 in rr._obj_geoms)):
                rr._obj_hits += 1
                obj_hit = True
            if ((g1 in rr._drone_geoms and g2 in rr._gate_geoms)
                    or (g2 in rr._drone_geoms
                        and g1 in rr._gate_geoms)):
                rr._gate_hits += 1


def run_pick(i):
    obj, start, alt, tgt = r.reset_scene_v2()
    task = A.PROMPT_MANIP.format(obj=obj)
    plat = V2Platform(r, start, 0.0)
    adr = r.cur["adr"]
    miss = 1e9
    lifted = False
    frames_n = 0
    traj = []
    vid = (imageio.get_writer("v2vid_%s_pick%02d.mp4" % (TAG, i),
                              fps=10, codec="libx264", quality=8,
                              macro_block_size=1) if VIDEO else None)
    for chunk_i in range(PICK_TICKS // EXEC):   # 120 s cap regardless
        with torch.no_grad():                   # of exec horizon
            batch = PRE(obs_batch(task))
            chunk = policy.predict_action_chunk(batch)
            chunk = POST(chunk)[0].float().cpu().numpy()[:HORIZON]
        for a in chunk[:EXEC]:
            plat.tick(a)
            frames_n += 1
            if vid is not None and frames_n % 2 == 0:
                f2 = r.frame(task)
                vid.append_data(np.concatenate(
                    [f2["observation.images.camera1"],
                     f2["observation.images.camera2"],
                     f2["observation.images.camera3"]], axis=1))
            aim, _ = r.live_target()
            d = float(np.linalg.norm(r.jaws() - aim))
            miss = min(miss, d)
            oz = float(r.data.qpos[adr + 2])
            if oz > A.MAT_TOP + 0.12 and bool(
                    r.data.eq_active[r.weld]):
                lifted = True
            if frames_n % 3 == 0:
                traj.append([round(float(x), 3) for x in
                             (*r.data.qpos[0:3], plat.yaw, *r.jaws())])
    if vid is not None:
        vid.close()
    ended_welded = bool(r.data.eq_active[r.weld])
    # episode-end cleanup (review 2026-09-04, pairing lens): an episode
    # that ends still CARRYING leaves the payload feed-forward latched
    # on nominal_hover_thrust -- every later episode would fly with a
    # wrong hover trim, keyed to this checkpoint's behaviour, which
    # breaks the paired cross-checkpoint comparison. Undo it while
    # r.cur is still this episode's object.
    if r._ff_on:
        r.weld_grasp(False)
    oxy = r.data.qpos[adr:adr + 2]
    d_bin = float(np.linalg.norm(oxy - r.bin_xy))
    placed = bool(d_bin <= 0.15
                  and float(r.data.qpos[adr + 2]) < A.MAT_TOP + 0.25)
    res = dict(kind="pick", ep=i, obj=obj, tag=TAG,
               miss_mm=round(miss * 1000, 1), picked=bool(lifted),
               placed=placed, success=placed,
               d_bin_mm=round(d_bin * 1000, 1),
               table_hits=r._table_hits, obj_hits=r._obj_hits,
               weld_tick=plat.weld_tick, ended_welded=ended_welded,
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
    yaw0 = float(2 * np.arctan2(r.data.qpos[6], r.data.qpos[3]))
    plat = V2Platform(r, start, yaw0)
    adr = r.cur["adr"]
    gx = r._gate_x
    crossed = False
    hover_run = 0
    hover_ok = False
    y_prev = float(r.data.qpos[1])
    frames_n = 0
    vid = (imageio.get_writer("v2vid_%s_nav%02d.mp4" % (TAG, i),
                              fps=10, codec="libx264", quality=8,
                              macro_block_size=1) if VIDEO else None)
    for chunk_i in range(NAV_TICKS // EXEC):    # 50 s cap regardless
        with torch.no_grad():                   # of exec horizon
            batch = PRE(obs_batch(task))
            chunk = policy.predict_action_chunk(batch)
            chunk = POST(chunk)[0].float().cpu().numpy()[:HORIZON]
        for a in chunk[:EXEC]:
            plat.tick(a, nav=True)
            frames_n += 1
            if vid is not None and frames_n % 2 == 0:
                f2 = r.frame(task)
                vid.append_data(np.concatenate(
                    [f2["observation.images.camera1"],
                     f2["observation.images.camera2"],
                     f2["observation.images.camera3"]], axis=1))
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
    if vid is not None:
        vid.close()
    return dict(kind="nav", ep=i, obj=obj, tag=TAG,
                crossed=bool(crossed), hover=bool(hover_ok),
                success=bool(crossed and hover_ok),
                gate_hits=r._gate_hits, table_hits=r._table_hits,
                obj_hits=r._obj_hits, frames=frames_n)


results = []
for i in range(N_PICK):
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
