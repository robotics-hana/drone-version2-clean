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
      [--rtc]       # E2: RTC prefix guidance across chunks (needs
                    # --exec < 50); default off = baseline sampling
      [--assist R]  # H1 hybrid: scripted terminal servo takes the
                    # last R metres to the weld (default 0 = off)
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
from platform_v2 import V2Platform
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
# E2 (pre-registered 2026-09-05): Real-Time Chunking prefix guidance
# during flow sampling (LeRobot's built-in RTCProcessor). E1 showed
# naive fast replanning is harmful (plan churn from fresh noise per
# inference); RTC conditions each new chunk on the previous chunk's
# unexecuted tail so consecutive plans stay consistent. Synchronous
# harness => inference_delay=0 (soft consistency over the overlap,
# nothing hard-frozen). Default off => every non-RTC path untouched.
# H1 (pre-registered 2026-09-07): terminal-servo handoff radius in
# metres -- when the POLICY brings the jaws this close to the target,
# the platform's scripted servo (collector-validated creep law) flies
# the final leg to the weld, then returns control. 0 = off; hybrid
# results are ALWAYS reported as their own rung, never as pure policy.
ASSIST = (float(sys.argv[sys.argv.index("--assist") + 1])
          if "--assist" in sys.argv else 0.0)
assert 0.0 <= ASSIST <= 0.30, "assist radius sanity bound"
RTC = "--rtc" in sys.argv
if RTC:
    assert EXEC < HORIZON, "--rtc needs --exec < %d (chunk overlap)" % HORIZON
KEY = {"observation.images.camera3": "observation.images.base_0_rgb",
       "observation.images.camera1": "observation.images.left_wrist_0_rgb",
       "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

policy = PI0Policy.from_pretrained(CKPT)
if RTC:
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    policy.config.rtc_config = RTCConfig(enabled=True,
                                         execution_horizon=EXEC)
    policy.init_rtc_processor()         # wires processor into model
torch.manual_seed(TORCHSEED)
torch.cuda.manual_seed_all(TORCHSEED)
PRE, POST = make_pre_post_processors(policy.config, pretrained_path=CKPT)
policy = policy.to("cuda").eval()


def infer_chunk(batch, prev_tail):
    """One policy inference. Returns (denormalized 50x7 numpy chunk,
    next prev_tail). Without --rtc this is EXACTLY the baseline call
    chain; with it, the previous chunk's unexecuted tail guides the
    flow sampling (prefix consistency). The tail is kept in the
    model's NORMALIZED action space and zero-padded to
    max_action_dim -- the space RTCProcessor compares against x_t.
    (Zero-padding matches LeRobot's own reference RTC rollout, which
    passes the unpadded normalized tail and lets the processor
    zero-pad identically; the pad dims are unsupervised at training,
    review 2026-09-05, so guidance there is a benign nudge to 0.)"""
    if RTC:
        raw = policy.predict_action_chunk(
            batch, prev_chunk_left_over=prev_tail,
            inference_delay=0, execution_horizon=EXEC)
        tail = raw[:, EXEC:HORIZON, :].detach()
        pad = policy.config.max_action_dim - tail.shape[-1]
        prev_tail = torch.nn.functional.pad(tail, (0, pad))
    else:
        raw = policy.predict_action_chunk(batch)
    chunk = POST(raw)[0].float().cpu().numpy()[:HORIZON]
    return chunk, prev_tail

# hash the physics/scene dependencies too, not just this script --
# pairing across runs depends on them and cluster file drift is a
# documented hazard (review 2026-09-04)
DEPS = {}
for _m in ("collect_airvla", "collect_v2", "pd_flight",
           "collect_demos", "platform_v2"):
    _mod = sys.modules.get(_m)
    if _mod is not None and getattr(_mod, "__file__", None):
        DEPS[_m] = hashlib.sha256(
            open(_mod.__file__, "rb").read()).hexdigest()[:12]
print("PROV " + json.dumps(dict(
    script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
    dep_sha=DEPS, exec_horizon=EXEC, rtc=RTC,
    assist_r=ASSIST,
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
    plat = V2Platform(r, start, 0.0, assist_r=ASSIST)
    adr = r.cur["adr"]
    miss = 1e9
    lifted = False
    frames_n = 0
    traj = []
    vid = (imageio.get_writer("v2vid_%s_pick%02d.mp4" % (TAG, i),
                              fps=10, codec="libx264", quality=8,
                              macro_block_size=1) if VIDEO else None)
    prev_tail = None                    # RTC: previous chunk's tail
    for chunk_i in range(PICK_TICKS // EXEC):   # 120 s cap regardless
        with torch.no_grad():                   # of exec horizon
            batch = PRE(obs_batch(task))
            chunk, prev_tail = infer_chunk(batch, prev_tail)
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
               assisted=bool(plat.took_tick is not None),
               assist_tick=plat.took_tick,
               assist_ticks=plat.assist_ticks,
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
    prev_tail = None                    # RTC: previous chunk's tail
    for chunk_i in range(NAV_TICKS // EXEC):    # 50 s cap regardless
        with torch.no_grad():                   # of exec horizon
            batch = PRE(obs_batch(task))
            chunk, prev_tail = infer_chunk(batch, prev_tail)
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
