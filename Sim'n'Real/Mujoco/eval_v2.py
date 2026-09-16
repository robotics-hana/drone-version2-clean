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
      [--learned-servo actor.pt]  # E5: learned terminal controller
                    # in place of the scripted servo (needs --assist)
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
SCENE_SEED = (int(sys.argv[sys.argv.index("--sceneseed") + 1])
              if "--sceneseed" in sys.argv else 97000)
                                       # default 97000 = the frozen
                                       # family, disjoint from
                                       # collection (71000), v1 eval
                                       # (77000), probes (88000);
                                       # 99000 reserved for the SOLO
                                       # probe's fresh test set
# SOLO probe (Hana 2026-09-13): --solo evaluates pick episodes on
# scenes containing ONLY the commanded object (distractor parked
# off-scene by reset_scene_v2) — pure grasp competence with the
# target-selection confound physically removed. A NEW PROTOCOL ARM,
# never mixed into the frozen two-object ladder; solo scenes are
# mildly out-of-distribution (all training scenes have two objects).
SOLO = "--solo" in sys.argv
# PARAPHRASE OOD (Hana 2026-09-14, "lets que em"): --paraphrase swaps
# the single training prompt template for an UNSEEN wording, cycling
# deterministically by episode index (ep % 5) so runs are exactly
# reproducible. Every training episode used PROMPT_MANIP verbatim, so
# any drop under this flag measures instruction-wording brittleness.
# Additive: flag absent = frozen behaviour, byte-identical prompts.
PARAPHRASE = "--paraphrase" in sys.argv
PARA_SET = [
    "grab the {obj} and drop it into the wooden box",
    "pick the {obj} up and place it in the box",
    "put the {obj} into the wooden box",
    "lift the {obj} and carry it over to the wooden box",
    "take the {obj} and set it down inside the wooden box",
]
# V3-ARM (pre-registered 2026-09-13): --policy-arm hands the arm
# joints to the policy -- action dims 3,4 applied by the platform as
# per-tick joint deltas (clip +-0.06), REPLACING the phase-based
# q_travel/q_carry switching. For v3-trained checkpoints (airvla_v3
# relabel); with the flag absent every existing path is untouched.
POLICY_ARM = "--policy-arm" in sys.argv
assert not (POLICY_ARM and "--learned-servo" in sys.argv), \
    "--policy-arm + --learned-servo composition is not pre-registered"
# OOD LINEUP (pre-registered 2026-09-15, Hana: "do a full OOD
# experimental line up" — robustness of the best system):
#   --synonyms         OOD-N: unseen object NAMES in the canonical
#                      template, cycled ep%3 (tests noun-level
#                      grounding vs trained-token memorization)
#   --oodpos           OOD-P: target forced outside the trained
#                      lateral band (reset_scene_v2; own seed family)
#   --novel-distractor OOD-D: v1-era mustard bottle as never-trained
#                      clutter on the SAME frozen scenes (paired)
# All additive, default-off = frozen paths byte-identical.
SYNONYMS = "--synonyms" in sys.argv
SYN_SET = {
    "plush penguin": ["toy penguin", "stuffed penguin",
                      "blue plush bird"],
    "weight": ["dumbbell", "metal weight", "calibration weight"],
}
OODPOS = "--oodpos" in sys.argv
NOVELDIST = "--novel-distractor" in sys.argv
assert not (SYNONYMS and PARAPHRASE), "one prompt manipulation at a time"
assert not (SOLO and (OODPOS or NOVELDIST)), "not pre-registered"
# C1 COMPOSITIONAL probe (pre-registered 2026-09-10; run approved by
# Hana 2026-09-15): --comp turns the pick loop into COMPOSITE
# episodes -- nav spawn behind the gate, composite instruction (the
# concatenation of the two trained prompts, never seen in training),
# staged scoring (crossed / approached / picked / placed; success =
# crossed AND picked AND placed), 1700-tick budget. Zero-shot may
# legitimately be ~0 -- a null is reportable. Default off = every
# existing path byte-identical.
# NOVEL-TARGET probe (Hana 2026-09-16, "can you test if it would
# grasp the mustard bottle?"): --novel-target commands the v1-era
# mustard bottle — never a target in any v2-family training — on the
# OOD-D scenes (bottle placed by the same deterministic rule, scenes
# byte-identical to the OOD-D arm; ONLY the instruction differs).
# True grasp cannot be scored (no weld machinery / aperture window
# for the bottle — the weld exists because the 0.2-0.5 N physical
# grip is solver-unreliable), so scoring is APPROACH-level:
# flew-to-bottle, closest jaw-to-cap distance, which object was
# approached instead, plus any physical bottle displacement/lift.
NOVELTGT = "--novel-target" in sys.argv
NOVELTGT_PROMPT = "pick up the mustard bottle and put it in the wooden box"
BOTTLE_CAP_H = 0.191            # cap height above support (v1 D8a)
assert not (NOVELTGT and (SOLO or PARAPHRASE or SYNONYMS or OODPOS)), \
    "novel-target is a pure-policy approach probe"
assert not (NOVELTGT and "--assist" in sys.argv), \
    "servo machinery is keyed to trained objects; novel-target runs pure"
COMP = "--comp" in sys.argv
assert not (NOVELTGT and COMP), "one probe at a time"
COMP_PROMPT = ("fly through the gate and hover over the {obj}, "
               "then pick up the {obj} and put it in the wooden box")
COMP_TICKS = 1700
assert not (COMP and (SOLO or OODPOS or NOVELDIST or PARAPHRASE
                      or SYNONYMS or POLICY_ARM or "--exec" in sys.argv)), \
    "the comp probe composes with nothing except --assist/--learned-servo"
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
# E5 (pre-registered 2026-09-07, actor frozen + ledgered 2026-09-08):
# --learned-servo <actor.pt> swaps the takeover behaviour from the
# scripted creep law to the LEARNED terminal controller
# (platform_e5.V2PlatformLearned; obs via rl_env.obs_vec, the
# training env's own function). Requires --assist R > 0 -- the
# engagement trigger and give-back machinery are H1's, unchanged.
# Default absent => no new imports, every existing path untouched.
LSERVO = (sys.argv[sys.argv.index("--learned-servo") + 1]
          if "--learned-servo" in sys.argv else None)
if LSERVO is not None:
    assert ASSIST > 0.0, "--learned-servo requires --assist R > 0"
    from platform_e5 import V2PlatformLearned
# PAG ablation (pre-registered ledger precondition, run approved by
# Hana 2026-09-08): --naive-payload removes the payload feed-forward
# trim that V2Runner.weld_grasp adds to the controller's hover thrust
# at the weld instant (collect_v2.py L119-138). The weld itself is
# unchanged -- the runner's weld_grasp is rebound to the BASE
# A.Runner method, so the constraint seats but the controller stays
# tuned for the unloaded mass (the paper's "no payload-aware
# guidance" condition). No frozen file is edited; default absent =>
# every existing path untouched.
NAIVE_PAYLOAD = "--naive-payload" in sys.argv
# TRUE-PAG arm (registered 2026-09-08 as the redesigned experiment
# after the trim falsification; built 2026-09-15): --pag updates the
# PD's total_mass by the payload's subtree mass at the weld instant
# (and back at release) -- the variable the PD's thrust AND arm
# gravity-moment feed-forwards actually read (the falsified trim
# bumped only the orphaned nominal_hover_thrust). Paired vs the
# shipped stack on identical scenes = what genuine payload-aware
# control buys (carry sag / settle / place accuracy from traj logs).
PAG = "--pag" in sys.argv
assert not (PAG and NAIVE_PAYLOAD), "pick one payload condition"
RTC = "--rtc" in sys.argv
if RTC:
    assert EXEC < HORIZON, "--rtc needs --exec < %d (chunk overlap)" % HORIZON
# Baseline-policy support (pre-registered 2026-09-10): --policy act
# evaluates a from-scratch ACT checkpoint on the identical frozen
# protocol. ACT trained on the dataset's NATIVE camera keys (no
# pretrained slot names), so the key map collapses to identity; ACT
# has no RTC and ignores the task string (language-blind by
# construction -- the stated point of the baseline). Default "pi0"
# leaves every existing path byte-identical.
PTYPE = (sys.argv[sys.argv.index("--policy") + 1]
         if "--policy" in sys.argv else "pi0")
assert PTYPE in ("pi0", "act", "diffusion"), PTYPE
if PTYPE != "pi0":
    assert not RTC, "--rtc is a pi0 flag; keep baselines pure"
    assert LSERVO is None, "--learned-servo not composed with baselines"
KEY = ({"observation.images.camera3": "observation.images.base_0_rgb",
        "observation.images.camera1": "observation.images.left_wrist_0_rgb",
        "observation.images.camera2": "observation.images.right_wrist_0_rgb"}
       if PTYPE == "pi0" else
       {"observation.images.camera1": "observation.images.camera1",
        "observation.images.camera2": "observation.images.camera2",
        "observation.images.camera3": "observation.images.camera3"})

if PTYPE == "act":
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(CKPT)
elif PTYPE == "diffusion":
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionPolicy)
    policy = DiffusionPolicy.from_pretrained(CKPT)
else:
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
           "collect_demos", "platform_v2",
           "platform_e5", "rl_env", "rl_nets"):
    _mod = sys.modules.get(_m)
    if _mod is not None and getattr(_mod, "__file__", None):
        DEPS[_m] = hashlib.sha256(
            open(_mod.__file__, "rb").read()).hexdigest()[:12]
print("PROV " + json.dumps(dict(
    script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
    dep_sha=DEPS, exec_horizon=EXEC, rtc=RTC,
    assist_r=ASSIST, learned_servo=LSERVO,
    naive_payload=NAIVE_PAYLOAD, policy_type=PTYPE, solo=SOLO,
    paraphrase=PARAPHRASE, policy_arm=POLICY_ARM,
    synonyms=SYNONYMS, oodpos=OODPOS, novel_distractor=NOVELDIST,
    comp=COMP, pag=PAG, novel_target=NOVELTGT,
    actor_sha=(hashlib.sha256(open(LSERVO, "rb").read())
               .hexdigest()[:12] if LSERVO else None),
    ckpt=CKPT, argv=sys.argv[1:], torch_seed=TORCHSEED,
    scene_seed=SCENE_SEED, tag=TAG,
    when=time.strftime("%Y-%m-%dT%H:%M:%S"))), flush=True)

r = V.V2Runner(SCENE_SEED)
if PAG:
    _weld0 = r.weld_grasp

    def _pag_weld(on):
        pre = r._ff_on
        _weld0(on)
        sub = float(r.model.body_subtreemass[r.cur["body"]])
        if on and not pre:
            r.ctrl.mppi.total_mass += sub
        elif (not on) and pre:
            r.ctrl.mppi.total_mass -= sub
    r.weld_grasp = _pag_weld
    print("PAG: total_mass payload update ACTIVE", flush=True)
if NAIVE_PAYLOAD:
    # bypass V2Runner's feed-forward override: weld seats, trim never
    # engages, _ff_on stays False (so end-of-episode FF cleanup is a
    # no-op by construction)
    r.weld_grasp = lambda on: A.Runner.weld_grasp(r, on)
    print("NAIVE-PAYLOAD: payload feed-forward DISABLED", flush=True)
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
    if PTYPE == "diffusion":
        # DP consumes n_obs_steps=2 stacked observations [B, T, ...];
        # keep a 1-frame history per key, duplicated at episode start
        # (cleared by run_pick/run_nav via _dp_hist.clear()). pi0/act
        # paths untouched.
        for k in list(batch.keys()):
            if k == "task":
                continue
            prev = _dp_hist.get(k, batch[k])
            _dp_hist[k] = batch[k]
            batch[k] = torch.stack([prev, batch[k]], dim=1)
    return batch


_dp_hist = {}


def run_pick(i):
    _dp_hist.clear()                     # fresh obs history per episode
    obj, start, alt, tgt = r.reset_scene_v2(
        solo=SOLO, oodpos=OODPOS,
        novel_distractor=(NOVELDIST or NOVELTGT))
    if NOVELTGT:
        task = NOVELTGT_PROMPT
        _madr = r.model.joint("mustard_free_joint").qposadr[0]
        _bx0 = np.array(r.data.qpos[_madr:_madr + 3])
    elif SYNONYMS:
        task = A.PROMPT_MANIP.format(obj=SYN_SET[obj][i % 3])
    else:
        task = (PARA_SET[i % len(PARA_SET)] if PARAPHRASE
                else A.PROMPT_MANIP).format(obj=obj)
    if LSERVO is not None:
        plat = V2PlatformLearned(r, start, 0.0, assist_r=ASSIST,
                                 actor_path=LSERVO)
    else:
        plat = V2Platform(r, start, 0.0, assist_r=ASSIST,
                          policy_arm=POLICY_ARM)
    adr = r.cur["adr"]
    miss = 1e9
    lifted = False
    miss_bottle = 1e9
    miss_other = 1e9
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
            if NOVELTGT:
                cap = np.array(r.data.qpos[_madr:_madr + 3]) \
                    + np.array([0, 0, 0.12])
                miss_bottle = min(miss_bottle, float(
                    np.linalg.norm(r.jaws() - cap)))
                other = [o for k, o in r.objs.items() if k != obj][0]
                miss_other = min(miss_other, float(np.linalg.norm(
                    r.jaws()[0:2]
                    - r.data.qpos[other["adr"]:other["adr"] + 2])))
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
               frames=frames_n,
               **({"prompt": task} if (PARAPHRASE or SYNONYMS) else {}),
               **({"kind_override": "ntgt",
                   "miss_bottle_mm": round(miss_bottle * 1000, 1),
                   "flew_bottle": bool(miss_bottle < 0.300),
                   "miss_trained_cmd_mm": round(miss * 1000, 1),
                   "miss_trained_other_mm": round(miss_other * 1000, 1),
                   "bottle_moved_mm": round(1000 * float(np.linalg.norm(
                       np.array(r.data.qpos[_madr:_madr + 2])
                       - _bx0[0:2])), 1),
                   "bottle_lifted": bool(
                       float(r.data.qpos[_madr + 2]) - _bx0[2] > 0.08)}
                  if NOVELTGT else {}))
    with open("eval_v2_traj.jsonl", "a") as fh:
        fh.write(json.dumps(dict(tag=TAG, kind="pick", ep=i, obj=obj,
                                 tgt=[round(float(x), 3) for x in tgt],
                                 bin_xy=[round(float(x), 3)
                                         for x in r.bin_xy],
                                 traj=traj)) + "\n")
    return res


def run_comp(i):
    """C1 composite episode: gate -> commanded object -> box, one
    instruction, staged scoring. Structure mirrors run_pick with the
    nav spawn/crossing tracking of run_nav; assist/learned-servo
    compose exactly as in picks (the servo is phase-agnostic)."""
    _dp_hist.clear()
    obj, start, alt, tgt = r.reset_scene_v2(comp=True)
    task = COMP_PROMPT.format(obj=obj)
    yaw0 = float(2 * np.arctan2(r.data.qpos[6], r.data.qpos[3]))
    if LSERVO is not None:
        plat = V2PlatformLearned(r, start, yaw0, assist_r=ASSIST,
                                 actor_path=LSERVO)
    else:
        plat = V2Platform(r, start, yaw0, assist_r=ASSIST)
    adr = r.cur["adr"]
    gx = r._gate_x
    crossed = False
    y_prev = float(r.data.qpos[1])
    miss = 1e9
    lifted = False
    frames_n = 0
    traj = []
    vid = (imageio.get_writer("v2vid_%s_comp%02d.mp4" % (TAG, i),
                              fps=10, codec="libx264", quality=8,
                              macro_block_size=1) if VIDEO else None)
    prev_tail = None
    for chunk_i in range(COMP_TICKS // EXEC):
        with torch.no_grad():
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
            y = float(r.data.qpos[1])
            if (y_prev < -0.6 <= y
                    and abs(float(r.data.qpos[0]) - gx) < 0.45
                    and 0.38 < float(r.data.qpos[2]) < 1.44):
                crossed = True
            y_prev = y
            aim, _ = r.live_target()
            miss = min(miss, float(np.linalg.norm(r.jaws() - aim)))
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
    if r._ff_on:
        r.weld_grasp(False)
    oxy = r.data.qpos[adr:adr + 2]
    d_bin = float(np.linalg.norm(oxy - r.bin_xy))
    placed = bool(d_bin <= 0.15
                  and float(r.data.qpos[adr + 2]) < A.MAT_TOP + 0.25)
    res = dict(kind="comp", ep=i, obj=obj, tag=TAG,
               crossed=bool(crossed), approached=bool(miss < 0.300),
               miss_mm=round(miss * 1000, 1), picked=bool(lifted),
               placed=placed,
               success=bool(crossed and lifted and placed),
               d_bin_mm=round(d_bin * 1000, 1),
               gate_hits=r._gate_hits, table_hits=r._table_hits,
               obj_hits=r._obj_hits, weld_tick=plat.weld_tick,
               ended_welded=ended_welded,
               assisted=bool(plat.took_tick is not None),
               assist_tick=plat.took_tick,
               assist_ticks=plat.assist_ticks, frames=frames_n)
    with open("eval_v2_traj.jsonl", "a") as fh:
        fh.write(json.dumps(dict(tag=TAG, kind="comp", ep=i, obj=obj,
                                 tgt=[round(float(x), 3) for x in tgt],
                                 bin_xy=[round(float(x), 3)
                                         for x in r.bin_xy],
                                 traj=traj)) + "\n")
    return res


def run_nav(i):
    _dp_hist.clear()                     # fresh obs history per episode
    obj, start, alt, tgt = r.reset_scene_v2(nav=True)
    task = A.PROMPT_NAV.format(obj=obj)
    yaw0 = float(2 * np.arctan2(r.data.qpos[6], r.data.qpos[3]))
    plat = V2Platform(r, start, yaw0, policy_arm=POLICY_ARM)
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
    res = run_comp(i) if COMP else run_pick(i)
    results.append(res)
    print("EVAL " + json.dumps(res), flush=True)
for i in range(N_NAV):
    res = run_nav(i)
    results.append(res)
    print("EVAL " + json.dumps(res), flush=True)

if NOVELTGT:
    nt = [x for x in results if x["kind"] == "pick"]
    if nt:
        mb = sorted(x["miss_bottle_mm"] for x in nt)
        print("NTGT SUMMARY: n=%d flew-to-bottle %d  bottle-median "
              "%.1f mm  bottle-moved %d  bottle-lifted %d  "
              "went-to-trained-instead %d"
              % (len(nt), sum(x["flew_bottle"] for x in nt),
                 mb[len(mb) // 2],
                 sum(x["bottle_moved_mm"] > 50 for x in nt),
                 sum(x["bottle_lifted"] for x in nt),
                 sum((not x["flew_bottle"])
                     and min(x["miss_trained_cmd_mm"],
                             x["miss_trained_other_mm"]) < 300
                     for x in nt)), flush=True)
comps = [x for x in results if x["kind"] == "comp"]
if comps:
    print("COMP SUMMARY: n=%d crossed %d approached %d picked %d "
          "placed %d SUCCESS %d"
          % (len(comps), sum(x["crossed"] for x in comps),
             sum(x["approached"] for x in comps),
             sum(x["picked"] for x in comps),
             sum(x["placed"] for x in comps),
             sum(x["success"] for x in comps)), flush=True)
picks = [x for x in results if x["kind"] == "pick"]
if picks:
    mm = sorted(x["miss_mm"] for x in picks)
    print("PICK SUMMARY: n=%d median %.1f mm  picked %d  placed %d"
          % (len(picks), mm[len(mm) // 2],
             sum(x["picked"] for x in picks),
             sum(x["placed"] for x in picks)), flush=True)
    # Target-true metrics (Hana 2026-09-11): the general median mixes
    # approach precision with target selection -- an episode that
    # flies to the DISTRACTOR contributes its (large) distance to the
    # commanded object. Restricting to episodes that actually went
    # for the commanded target (miss < 300 mm, below the object-
    # separation band -- the established wrong-object criterion)
    # separates the two: target-true median = how close it gets WHEN
    # it goes for the right object; flew-to-target = how often it
    # does. Additive print only; nothing upstream changes.
    tt = [x for x in mm if x < 300]
    if tt:
        print("TARGET-TRUE: flew-to-target %d/%d  median %.1f mm"
              % (len(tt), len(picks), tt[len(tt) // 2]), flush=True)
    else:
        print("TARGET-TRUE: flew-to-target 0/%d  median n/a"
              % len(picks), flush=True)
navs = [x for x in results if x["kind"] == "nav"]
if navs:
    print("NAV SUMMARY: n=%d success %d/%d"
          % (len(navs), sum(x["success"] for x in navs), len(navs)),
          flush=True)
print("EVALV2-DONE", flush=True)
