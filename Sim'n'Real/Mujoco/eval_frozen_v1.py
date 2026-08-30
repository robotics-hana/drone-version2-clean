"""Phase E eval harness: pi0 closed-loop in the collection sim, all three
of the paper's inference methods (doc section 7):

  (a) naive   -- chunk H=50, execute all, replan at chunk boundaries
  (b) rtc     -- Real-Time Chunking: execution horizon 25, exponential
                 prefix-attention schedule, 10 denoising steps, simulated
                 inference latency (LeRobot's built-in RTCProcessor)
  (c) guided  -- rtc + Payload-Aware Guidance: v_guid = v - s(tau)*xi,
                 xi = grad_A Phi(A_hat) under the reconstruction-guidance
                 approximation grad_x A_hat ~= I (doc D7); Phi_payload =
                 (lambda_z/2) * alpha * sum_t w_t (z_t - z_des)^2 with
                 z_des = z_curr + 0.15 m, w_t = (t/(H-1))^1, alpha from
                 close-intent over the last K=4 grip commands fused (max)
                 with measured aperture. s(tau) constant, CLI-tunable
                 (doc D6: the paper never specifies it).

The ARM is platform-owned (D4): a deterministic automaton of commanded
grip, weld state and commanded altitude -- never the object's pose.
The WELD engages platform-side on a measured seated pinch; an air-weld
cannot fake success because scoring checks the object's final pose.

Usage:
  eval_pi0.py stub                              # plumbing dry-run
  eval_pi0.py apitest CKPT [--mode M] [cpu]     # one-chunk API check
  eval_pi0.py CKPT [n_pick n_nav] [--mode naive|rtc|guided]
              [--delay 2] [--s 1.0] [cpu]
"""
import json
import sys

import mujoco
import numpy as np

import collect_airvla as A
import collect_demos as C

MAT_TOP = A.MAT_TOP
PROMPT_TRAIN = "pick up the {obj} and put it in the wooden box"
PROMPT_HELDOUT = "put the {obj} in the wooden box"
PROMPT_NAV = "fly through the gate and hover over the {obj}"
# compositional prompt (paper section IV): concatenation of the two
# trained task instructions -- NEVER seen in training; stages must
# complete in order and grasp-before-gate is a failure
PROMPT_COMP = ("fly through the gate and hover over the {obj}, then "
               "pick up the {obj} and put it in the wooden box")

CHUNK = 50            # pi0 chunk size (paper H=50)
EXEC_H = 25           # RTC execution horizon (paper: H=25)
RTC_STEPS = 10        # RTC denoising steps (paper)
LAMBDA_Z = 0.5        # payload guidance weight (paper)
DZ_DES = 0.15         # desired altitude gain after grasp intent (paper)
K_INTENT = 4          # grip-command window for close intent (paper)

PRE = POST = None     # checkpoint-shipped processor pipelines
VID_EPS = 5           # pick episodes recorded per run (user request:
_BANNERS = {}         # 5-episode 3-camera review of the guided mode)
TRAJ_PATH = "/home/ucabhe0/Scratch/airvla/evalout/eval_traj.jsonl"   # per-episode drone/
MODE_TAG = ["?"]
RUN_TAG = ["?"]
# Eval-side leash on the setpoint integrator (diagnostic, NOT a fix). sp is a
# hidden integrator: Platform.tick accumulates the policy's deltas with no
# feedback from measured pose, and the policy only ever observes qpos -- so a
# biased dz/dy runs sp away unbounded (measured: y 1.07 -> 3.46 m in one
# rollout). During COLLECTION the expert forms deltas as err = goal - sp, which
# is self-limiting, so this divergence cannot occur in the training data.
# Clamping sp to within N metres of the true pose tests whether the runaway is
# the cause of the failure or merely a symptom of it. 0 = original behaviour.
SP_CLAMP = [0.0]
# Pick START-STATE fix. The collector spawns picks on the DECK --
# rng.uniform(0.21, 0.30) -- because the profile gains altitude vertically at
# the spawn (up-then-across takeoff, D42) and never sheds it en route. It also
# REJECTION-SAMPLES: >= 0.70 m from the object and room-side of the table front
# edge (anchor_y + 0.35). eval_pi0 was never updated and starts picks mid-air at
# U(0.55, 0.95) with no rejection, so ZERO of 225 training pick episodes share
# an initial altitude with any eval episode -- the policy is off-distribution
# before it acts. Nav is unaffected (161/175 overlap) and nav is the task that
# works. 0 = original behaviour, for A/B against the ladder baseline.
START_FIX = [0]
# Platform repair (fixfit5, 2026-08-30: ground-truth replay A=3/3 B=3/3).
# Root cause chain: the stale sp[2]<0.42 arm gate held q_travel at the
# table-era grasp altitude (z~0.53), parking the jaws 100-150 mm off; the
# aperture-window weld then fired on air and TOWED objects. Repair:
#   clip 0.035 (matches collector SP_STEP; old 0.03 truncated the top 14%)
#   weld = pad-object CONTACT (debounced 2-of-4; plush contact flickers)
#          + seated aperture (0.004, 0.0170) -- air-welds impossible
#   arm  = travel -> deploy (first sp_z>=0.50) -> tuck (weld+60)
#          -> travel (release), poses CALIBRATED from training demos
#          (system identification, not runtime privilege); slewed 0.06/tick.
# kind-gated: nav episodes keep q_travel throughout, as in collection.
PLATFIX = [0]
# Diagnostic flags (2026-08-30 reviewer round):
# EXECH: shorten the naive execution horizon (replan every N actions instead
#   of 50). Tests whether late-arriving wrist information cannot be acted on
#   between 5 s open-loop replans -- a control-frequency mechanism that looks
#   identical to lateral blindness from the outside.
# ORACLEYAW: platform pins commanded heading at the bearing to the object
#   (privileged; diagnostic only). Directly tests the reinstated yaw
#   mechanism: far-field heading-error sign anti-predicts lateral-miss sign
#   in 18/20 episodes (binom p~2e-4). If lateral scatter collapses under
#   oracle heading, the fault is the aiming channel, not perception.
EXECH = [0]
ORACLEYAW = [0]
# ORACLESTATE: append the true jaws-to-aim vector to observation.state
# (10 -> 13 dims) to evaluate a policy fine-tuned on the oracle dataset.
# Privileged; diagnostic only -- bounds the problem from above.
ORACLESTATE = [0]
# ORACLECORRUPT: attendance control for the oracle ablation. The true
# jaws-to-aim vector is rotated about z by a fixed per-episode random angle:
# magnitude and vertical component preserved (distributionally plausible),
# lateral direction uninformative. If the oracle-trained policy behaves the
# same with corrupted vectors, it never learned to attend to the feature and
# an oracle null says nothing about perception; if it degrades, the feature
# is load-bearing and a null becomes a real finding.
ORACLECORRUPT = [0]
CORRUPT_ANGLE = [0.0]
# Reproducibility (frozen-protocol requirements, 2026-08-30):
# pi0 is a flow-matching model -- it SAMPLES noise at every inference, so an
# unseeded run is unrepeatable even with identical checkpoint, scene and
# flags. TORCHSEED seeds torch (cpu+cuda) right after the policy loads and
# is recorded in the PROV line, which is printed first and carries script
# sha, checkpoint, argv, seeds and timestamp for every results file.
TORCHSEED = [1000]
PF_Q_TRAVEL = np.array([-0.053, 0.267])
PF_Q_DEPLOY = np.array([-1.333, 1.547])
PF_Q_TUCK = np.array([0.533, -0.160])      # object trajectories for the results dashboard


def banner(task, width):
    if task not in _BANNERS:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, 46), (12, 12, 12))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                26)
        except OSError:
            f = ImageFont.load_default()
        d.text((10, 8), task, font=f, fill=(255, 255, 255))
        _BANNERS[task] = np.asarray(img).copy()
    return _BANNERS[task]


def compose_frame(r, task):
    """wrist | forward | external tile row with the task banner."""
    tiles = []
    for cam in (A.CAMS["camera1"], A.CAMS["camera2"], "lab_external"):
        r.rend.update_scene(r.data, camera=cam)
        tiles.append(r.rend.render().copy())
    fr = np.concatenate(tiles, axis=1)
    b = banner(task, fr.shape[1])
    fr[0:b.shape[0]] = b
    return fr


def arg_after(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


class Platform:
    """Platform-side setpoint integrator + arm/weld automaton (D4)."""

    def __init__(self, r, start, yaw):
        self.r, self.ex = r, r.expert
        self.sp = np.array(start, float)
        self.yaw = float(yaw)
        self.grip = 1.0
        self.ap_hold = 0
        self.kind = "pick"
        m = r.model
        ob = r.cur["body"]
        self.pf_obj = set(g for g in range(m.ngeom)
                          if m.body_rootid[m.geom_bodyid[g]] == ob)
        self.pf_pads = {m.geom("pad_1").id, m.geom("pad_2").id}
        self.pf_hist = []
        self.pf_weld_tick = None
        self.pf_released = False
        self.pf_deployed = False
        self.pf_i = 0
        self.pf_cmd = PF_Q_TRAVEL.copy()

    def _tick_fixed(self, act):
        """fixfit5-validated path (see PLATFIX above)."""
        r = self.r
        self.sp = self.sp + np.clip(act[0:3], -0.035, 0.035)
        self.yaw += float(np.clip(act[5], -0.06, 0.06))
        self.grip = float(np.clip(act[6], 0.0, 1.0))
        welded = bool(r.data.eq_active[r.weld])
        ap = float(r.data.qpos[r.gadr])
        hit = False
        d = r.data
        for ci in range(d.ncon):
            g1 = int(d.contact.geom1[ci])
            g2 = int(d.contact.geom2[ci])
            if (g1 in self.pf_pads and g2 in self.pf_obj) or \
               (g2 in self.pf_pads and g1 in self.pf_obj):
                hit = True
                break
        if not welded and 0.004 < ap < 0.0170:
            self.pf_hist = (self.pf_hist + [hit])[-4:]
            if sum(self.pf_hist) >= 2:
                r.weld_grasp(True)
        elif welded and self.grip > 0.8:
            r.weld_grasp(False)
            self.pf_hist = []
        else:
            self.pf_hist = []
        welded = bool(r.data.eq_active[r.weld])
        if welded and self.pf_weld_tick is None:
            self.pf_weld_tick = self.pf_i
        if self.pf_weld_tick is not None and not welded and self.grip > 0.8:
            self.pf_released = True
        if not self.pf_deployed:
            # pick: deploy on the takeoff climb (validated fixfit5 path).
            # comp: the nav leg flies at 0.60-1.00 with the arm in travel, as
            # nav training does; deploy only on the descent to approach
            # altitude, so the arm is never down through the gate.
            if self.kind == "comp":
                if self.sp[2] <= 0.58:
                    self.pf_deployed = True
            elif self.sp[2] >= 0.50:
                self.pf_deployed = True
        if self.kind == "nav" or not self.pf_deployed or self.pf_released:
            tgt = PF_Q_TRAVEL
        elif (self.pf_weld_tick is not None
              and self.pf_i >= self.pf_weld_tick + 60):
            tgt = PF_Q_TUCK
        else:
            tgt = PF_Q_DEPLOY
        self.pf_cmd = self.pf_cmd + np.clip(tgt - self.pf_cmd, -0.06, 0.06)
        self.pf_i += 1
        if ORACLEYAW[0] and self.kind != "nav":
            v = r.data.qpos[r.cur["adr"]:r.cur["adr"] + 2] - r.data.qpos[0:2]
            if float(np.linalg.norm(v)) > 0.05:
                # nose = body -y: nose_world = (sin yaw, -cos yaw) = v_hat
                self.yaw = float(np.arctan2(v[0], -v[1]))
        r.ctrl.set_targets(self.sp, self.pf_cmd,
                           C.GRIPPER_OPEN * self.grip)
        r.ctrl.mppi.target_yaw = self.yaw
        for _ in range(r.sub):
            r.ctrl.step()
            mujoco.mj_step(r.model, r.data)

    def tick(self, act):
        if PLATFIX[0]:
            return self._tick_fixed(act)
        r, ex = self.r, self.ex
        self.sp = self.sp + np.clip(act[0:3], -0.03, 0.03)
        if SP_CLAMP[0] > 0:
            off = self.sp - r.data.qpos[0:3]
            n = float(np.linalg.norm(off))
            if n > SP_CLAMP[0]:
                self.sp = r.data.qpos[0:3] + off * (SP_CLAMP[0] / n)
        self.yaw += float(np.clip(act[5], -0.06, 0.06))
        self.grip = float(np.clip(act[6], 0.0, 1.0))
        welded = bool(r.data.eq_active[r.weld])
        ap = float(r.data.qpos[r.gadr])
        if not welded and self.grip < 0.5 and 0.005 < ap < 0.0135:
            self.ap_hold += 1
            if self.ap_hold >= 3:
                r.weld_grasp(True)
        else:
            self.ap_hold = 0
        if welded and self.grip > 0.8:
            r.weld_grasp(False)
        if welded:
            q = ex.q_carry
        elif self.sp[2] < 0.42:
            q = ex.q_grasp
        else:
            q = ex.q_travel
        r.ctrl.set_targets(self.sp, q, C.GRIPPER_OPEN * self.grip)
        r.ctrl.mppi.target_yaw = self.yaw
        for _ in range(r.sub):
            r.ctrl.step()
            mujoco.mj_step(r.model, r.data)


def obs_batch(r, task, torch, dev, dtype, keymap):
    f = {}
    for key, cam in A.CAMS.items():
        r.rend.update_scene(r.data, camera=cam)
        img = torch.from_numpy(r.rend.render().copy()).permute(2, 0, 1)
        f[keymap["observation.images." + key]] = (
            img.unsqueeze(0).to(dev).to(dtype) / 255.0)
    st_ = r.state()
    if ORACLESTATE[0]:
        aim_, _ = r.live_target()
        v_ = (np.asarray(aim_) - r.jaws()).astype(np.float32)
        if ORACLECORRUPT[0]:
            c_, s2_ = np.cos(CORRUPT_ANGLE[0]), np.sin(CORRUPT_ANGLE[0])
            v_ = np.array([c_ * v_[0] - s2_ * v_[1],
                           s2_ * v_[0] + c_ * v_[1], v_[2]], np.float32)
        st_ = np.concatenate([st_, v_])
    f["observation.state"] = torch.from_numpy(
        st_).unsqueeze(0).to(dev).to(dtype)
    f["task"] = task
    return f


# ---------------------------------------------------------------------------
# inference drivers, one per ladder rung
# ---------------------------------------------------------------------------
class NaiveDriver:
    """select_action with pi0's internal 50-action queue: execute the
    whole chunk, replan blind at the boundary."""

    def __init__(self, policy, torch, dev, dtype, keymap):
        self.p, self.t = policy, torch
        self.dev, self.dtype, self.keymap = dev, dtype, keymap

    def reset(self):
        self.p.reset()

    def act(self, r, task):
        batch = obs_batch(r, task, self.t, self.dev, self.dtype,
                          self.keymap)
        batch = PRE(batch)
        with self.t.no_grad():
            a = self.p.select_action(batch)
        a = POST(a)
        return a.squeeze(0).float().cpu().numpy()


class RTCDriver:
    """Chunked execution with Real-Time Chunking: after EXEC_H actions,
    re-infer conditioned on the unexecuted tail (prev_chunk_left_over)
    with a simulated inference latency of `delay` ticks. The synchronous
    simulation executes the new chunk from index 0 -- its first `delay`
    actions are the frozen copies of what would have executed while the
    GPU was busy, which is exactly RTC's contract."""

    def __init__(self, policy, torch, dev, dtype, keymap, delay,
                 unnorm):
        self.p, self.t = policy, torch
        self.dev, self.dtype, self.keymap = dev, dtype, keymap
        self.delay, self.unnorm = delay, unnorm
        self.norm = None      # torch (1, CHUNK, 7), model frame
        self.real = None      # np (CHUNK, 7), world units
        self.ptr = 0

    def reset(self):
        self.p.reset()
        self.norm, self.real, self.ptr = None, None, 0

    def _infer(self, r, task, leftover, delay):
        batch = obs_batch(r, task, self.t, self.dev, self.dtype,
                          self.keymap)
        batch = PRE(batch)
        with self.t.no_grad():
            chunk = self.p.predict_action_chunk(
                batch, inference_delay=delay,
                prev_chunk_left_over=leftover,
                execution_horizon=EXEC_H)
        self.norm = chunk
        self.real = self.unnorm(chunk)
        self.ptr = 0

    def act(self, r, task):
        if self.norm is None:
            self._infer(r, task, None, 0)
        elif self.ptr >= EXEC_H:
            leftover = self.norm[:, self.ptr:, :]
            self._infer(r, task, leftover, self.delay)
        a = self.real[self.ptr]
        self.ptr += 1
        return a


class GuidedDriver(RTCDriver):
    """RTC + Payload-Aware Guidance. Wraps the model's denoise_step to
    subtract s(tau)*xi; the guidance context (alpha, z_curr) refreshes
    at every inference from platform-observable signals only."""

    def __init__(self, *args, s_scale=1.0, act_scale=None, **kw):
        super().__init__(*args, **kw)
        self.s_scale = s_scale
        self.act_scale = act_scale        # per-dim normalizer scale
        self.grip_hist = [1.0] * K_INTENT
        self.ctx = dict(alpha=0.0, z_curr=0.0)
        self._wrap()

    def _wrap(self):
        model, t = self.p.model, self.t
        orig = model.denoise_step
        ctx, s_scale = self.ctx, self.s_scale
        std_z = float(self.act_scale[2])

        def guided(state, prefix_pad_masks, past_key_values, x_t,
                   timestep):
            v = orig(state, prefix_pad_masks, past_key_values, x_t,
                     timestep)
            alpha = ctx["alpha"]
            if alpha <= 0.0:
                return v
            tau = float(timestep.reshape(-1)[0])
            a_hat = x_t - tau * v          # one-step estimate (t: 1->0)
            dz_real = a_hat[:, :, 2].float() * std_z
            z_traj = ctx["z_curr"] + t.cumsum(dz_real, dim=1)
            h = z_traj.shape[1]
            w = (t.arange(h, device=z_traj.device, dtype=z_traj.dtype)
                 / max(1, h - 1)) ** 1.0
            resid = w * (z_traj - (ctx["z_curr"] + DZ_DES))
            grad_real = LAMBDA_Z * alpha * t.flip(
                t.cumsum(t.flip(resid, [1]), dim=1), [1])
            xi = t.zeros_like(v)
            xi[:, :, 2] = (grad_real * std_z).to(v.dtype)
            return v - s_scale * xi

        model.denoise_step = guided

    def act(self, r, task):
        ap = float(r.data.qpos[r.gadr])
        intent = 1.0 - min(self.grip_hist)
        ap_closed = max(0.0, 1.0 - ap / 0.016)
        self.ctx["alpha"] = float(np.clip(max(intent, ap_closed), 0, 1))
        self.ctx["z_curr"] = float(r.data.qpos[2])
        a = super().act(r, task)
        self.grip_hist = self.grip_hist[1:] + [float(a[6])]
        return a


def make_unnorm(torch, dev):
    """Derive the action unnormalization affine numerically from the
    shipped POST pipeline (pipeline-agnostic: probe with 0s and 1s)."""
    z = POST(torch.zeros(1, CHUNK, 7, device=dev))
    o = POST(torch.ones(1, CHUNK, 7, device=dev))
    offset = z.float().cpu().numpy()[0]
    scale = o.float().cpu().numpy()[0] - offset

    def unnorm(chunk):
        return chunk.float().cpu().numpy()[0] * scale + offset
    return unnorm, scale[0]      # per-dim scale row for guidance


def in_box(r):
    adr = r.cur["adr"]
    p = r.data.qpos[adr:adr + 2] - r.bin_xy
    cy, sy = np.cos(-r.bin_yaw), np.sin(-r.bin_yaw)
    local = np.array([cy * p[0] - sy * p[1], sy * p[0] + cy * p[1]])
    z = float(r.data.qpos[adr + 2])
    return (abs(local[0]) < 0.13 and abs(local[1]) < 0.13
            and z < MAT_TOP + 0.12)


def run_episode(r, driver, kind, obj, prompt, vid=None):
    rng = r.rng
    bxy = np.array([rng.uniform(-0.60, 0.60), rng.uniform(0.20, 1.00)])
    if kind == "pick":
        if START_FIX[0]:
            anchor = bxy
            for _ in range(300):
                start = np.array([rng.uniform(-0.5, 0.5),
                                  rng.uniform(0.9, 1.6),
                                  rng.uniform(0.21, 0.30)])
                if (float(np.linalg.norm(start[0:2] - anchor)) >= 0.70
                        and start[1] >= anchor[1] + 0.35):
                    break
        else:
            start = np.array([rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.6),
                              rng.uniform(0.55, 0.95)])
        yaw0 = 0.0
        r.reset_scene(bxy, start, obj=obj)
        if "--objyaw0" in sys.argv:
            # yaw-dependence diagnostic: pin the object heading to 0 so
            # the grasp yaw ~ 0 and world-frame action deltas coincide
            # with the body frame. A pick-rate jump here implicates the
            # world-frame action convention (D3) in the descent failure.
            adr0 = r.cur["adr"]
            r.data.qpos[adr0 + 3:adr0 + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(r.model, r.data)
        limit = int(120 * A.FPS)
    else:
        gx = float(rng.choice([-0.7, 0.7]))
        start = np.array([rng.uniform(-0.95, 0.95),
                          rng.uniform(-1.9, -1.2), rng.uniform(0.60, 1.00)])
        yaw0 = np.pi + rng.uniform(-0.26, 0.26)
        r.reset_scene(bxy, start, gate_xy=np.array([gx, -0.6]),
                      yaw=yaw0, obj=obj)
        limit = int(45 * A.FPS) if kind == "nav" else int(180 * A.FPS)
    task = prompt.format(obj=obj)
    if ORACLECORRUPT[0]:
        CORRUPT_ANGLE[0] = float(rng.uniform(0.5, 2 * np.pi - 0.5))
    plat = Platform(r, start, yaw0)
    plat.kind = kind
    adr = r.cur["adr"]
    lifted = crossed = hover_done = wrong_order = False
    gate_hit = False   # paper: clipping a gate member is a crash --
    hover_run = 0      # the earlier scorer missed this (user caught it
    miss = 1e9         # in the review video, 2026-08-26)
    gate_geoms = set()
    drone_geoms = set()
    if kind != "pick":
        droot = r.model.body_rootid[r.model.body("base_link").id]
        for gi in range(r.model.ngeom):
            if r.model.geom_bodyid[gi] == r.gate:
                gate_geoms.add(gi)
            elif r.model.body_rootid[r.model.geom_bodyid[gi]] == droot:
                drone_geoms.add(gi)
    y_prev = float(r.data.qpos[1])
    traj = []
    t = 0
    for t in range(limit):
        if driver is None:                      # stub: drift forward
            act = np.array([0, 0.002, 0, 0, 0, 0, 1.0], np.float32)
        else:
            act = driver.act(r, task)
        plat.tick(act)
        if vid is not None and t % 2 == 0:
            if kind == "nav" and "--allcam3" not in sys.argv:
                # gate-task review: ROOM camera only (user request),
                # task banner kept
                r.rend.update_scene(r.data, camera="lab_external")
                fr = r.rend.render().copy()
                b = banner(task, fr.shape[1])
                fr[0:b.shape[0]] = b
                vid.append(fr)
            else:
                vid.append(compose_frame(r, task))
        if t % 3 == 0:
            qw_, qx_, qy_, qz_ = [float(v) for v in r.data.qpos[3:7]]
            yaw_ = float(np.arctan2(2 * (qw_ * qz_ + qx_ * qy_),
                                    1 - 2 * (qy_ * qy_ + qz_ * qz_)))
            jx_, jy_, jz_ = [float(v) for v in r.jaws()]
            traj.append([round(float(r.data.qpos[0]), 3),
                         round(float(r.data.qpos[1]), 3),
                         round(float(r.data.qpos[2]), 3),
                         round(yaw_, 3),
                         round(jx_, 3), round(jy_, 3), round(jz_, 3)])
        if gate_geoms and not gate_hit:
            for ci in range(r.data.ncon):
                c = r.data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if ((g1 in gate_geoms and g2 in drone_geoms)
                        or (g2 in gate_geoms and g1 in drone_geoms)):
                    gate_hit = True
                    break
        if kind != "nav":
            aim_now, _ = r.live_target()
            miss = min(miss, float(np.linalg.norm(r.jaws() - aim_now)))
        now_lifted = (bool(r.data.eq_active[r.weld])
                      and float(r.data.qpos[adr + 2]) > MAT_TOP + 0.12)
        if now_lifted and not lifted and kind == "comp" and not crossed:
            wrong_order = True     # paper: grasp-before-gate = failure
        lifted |= now_lifted
        y = float(r.data.qpos[1])
        if kind != "pick" and y_prev < -0.6 <= y:
            gp = r.model.body_pos[r.gate]
            if (abs(float(r.data.qpos[0]) - gp[0]) < 0.45
                    and 0.34 < float(r.data.qpos[2]) - gp[2] < 1.40):
                crossed = True
        y_prev = y
        if kind != "pick" and not hover_done:
            # doc section 4 hover criterion: within 0.25 m over the
            # object, sustained 3 s (30 ticks)
            near = float(np.linalg.norm(
                r.data.qpos[0:2] - r.data.qpos[adr:adr + 2])) < 0.25
            hover_run = hover_run + 1 if (crossed and near) else 0
            if hover_run >= 30:
                hover_done = True
                if kind == "nav":
                    break
        if kind != "nav" and t > 50 and in_box(r) and not bool(
                r.data.eq_active[r.weld]):
            break
    adr0 = r.cur["adr"]
    try:
        with open(TRAJ_PATH, "a") as f:
            f.write(json.dumps(dict(
                mode=MODE_TAG[0], tag=RUN_TAG[0], kind=kind, obj=obj,
                obj_xy=[round(float(r.data.qpos[adr0]), 3),
                        round(float(r.data.qpos[adr0 + 1]), 3)],
                bin_xy=[round(float(r.bin_xy[0]), 3),
                        round(float(r.bin_xy[1]), 3)],
                traj=traj)) + "\n")
    except OSError:
        pass
    if kind == "pick":
        placed = bool(in_box(r))
        return dict(kind=kind, obj=obj, prompt=prompt,
                    picked=bool(lifted), placed=placed, success=placed,
                    miss_mm=round(miss * 1000, 1), frames=int(t + 1))
    if kind == "comp":
        placed = bool(in_box(r))
        ok = bool(crossed and hover_done and lifted and placed
                  and not wrong_order and not gate_hit)
        return dict(kind=kind, obj=obj, prompt=prompt,
                    crossed=bool(crossed), gate_hit=bool(gate_hit),
                    hover=bool(hover_done),
                    picked=bool(lifted), placed=placed,
                    wrong_order=bool(wrong_order), success=ok,
                    miss_mm=round(miss * 1000, 1), frames=int(t + 1))
    return dict(kind=kind, obj=obj, prompt=prompt, crossed=bool(crossed),
                gate_hit=bool(gate_hit),
                clean_cross=bool(crossed and not gate_hit),
                success=bool(hover_done and not gate_hit),
                frames=int(t + 1))


def main():
    global PRE, POST
    r = A.Runner(77000)
    if sys.argv[1] == "stub":
        res = run_episode(r, None, "pick", "plush penguin", PROMPT_TRAIN)
        print("STUB", json.dumps(res), flush=True)
        res = run_episode(r, None, "nav", "weight", PROMPT_NAV)
        print("STUB", json.dumps(res), flush=True)
        print("PLUMBING-OK", flush=True)
        return

    import imageio.v2 as imageio
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    apitest = sys.argv[1] == "apitest"
    ckpt = sys.argv[2] if apitest else sys.argv[1]
    mode = arg_after("--mode", "naive")
    delay = int(arg_after("--delay", "2"))
    s_scale = float(arg_after("--s", "1.0"))
    SP_CLAMP[0] = float(arg_after("--spclamp", "0"))
    START_FIX[0] = 1 if "--startfix" in sys.argv else 0
    PLATFIX[0] = 1 if "--platfix" in sys.argv else 0
    EXECH[0] = int(arg_after("--exech", "0"))
    ORACLEYAW[0] = 1 if "--oracleyaw" in sys.argv else 0
    ORACLESTATE[0] = 1 if "--oraclestate" in sys.argv else 0
    ORACLECORRUPT[0] = 1 if "--oraclecorrupt" in sys.argv else 0
    TORCHSEED[0] = int(arg_after("--torchseed", "1000"))
    globals()["VID_EPS"] = int(arg_after("--videps", str(VID_EPS)))
    pos = [a for a in sys.argv[2:] if a.isdigit()]
    n_pick = int(pos[0]) if len(pos) > 0 else 6
    n_nav = int(pos[1]) if len(pos) > 1 else 3
    n_comp = int(pos[2]) if len(pos) > 2 else 0
    dev = "cpu" if "cpu" in sys.argv else "cuda"
    # applied after policy load below via config.n_action_steps
    dtype = torch.float32

    policy = PI0Policy.from_pretrained(ckpt)
    torch.manual_seed(TORCHSEED[0])
    torch.cuda.manual_seed_all(TORCHSEED[0])
    import hashlib, time as _time
    _prov = dict(
        script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12],
        ckpt=str(ckpt), argv=sys.argv[1:], torch_seed=TORCHSEED[0],
        runner_seed=77000, vid_eps=VID_EPS,
        when=_time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("PROV " + json.dumps(_prov), flush=True)
    if EXECH[0]:
        assert hasattr(policy.config, "n_action_steps"), "no n_action_steps"
        policy.config.n_action_steps = EXECH[0]
        print("EXECH: n_action_steps ->", policy.config.n_action_steps, flush=True)
    policy.to(dev).eval()
    PRE, POST = make_pre_post_processors(policy.config,
                                         pretrained_path=ckpt)
    keymap = {
        "observation.images.camera3": "observation.images.base_0_rgb",
        "observation.images.camera1": "observation.images.left_wrist_0_rgb",
        "observation.images.camera2": "observation.images.right_wrist_0_rgb"}

    if mode in ("rtc", "guided"):
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            execution_horizon=EXEC_H)
        policy.init_rtc_processor()
        policy.config.num_inference_steps = RTC_STEPS
    unnorm, act_scale = make_unnorm(torch, dev)
    if mode == "naive":
        driver = NaiveDriver(policy, torch, dev, dtype, keymap)
    elif mode == "rtc":
        driver = RTCDriver(policy, torch, dev, dtype, keymap, delay,
                           unnorm)
    else:
        driver = GuidedDriver(policy, torch, dev, dtype, keymap, delay,
                              unnorm, s_scale=s_scale,
                              act_scale=act_scale)
    MODE_TAG[0] = mode
    RUN_TAG[0] = arg_after("--tag", "untagged")
    print("MODE", mode, "delay", delay, "s", s_scale, flush=True)

    if apitest:
        r.reset_scene(np.array([0.0, 0.6]), np.array([0.0, 1.2, 0.75]),
                      obj="plush penguin")
        mujoco.mj_forward(r.model, r.data)
        driver.reset()
        for i in range(EXEC_H + 2):     # force one RTC re-inference
            act = driver.act(
                r, PROMPT_TRAIN.format(obj="plush penguin"))
        print("ACTION", np.round(act, 4).tolist(), flush=True)
        print("API-OK", flush=True)
        return

    vp = int(arg_after("--videps", str(VID_EPS)))
    allcam3 = "--allcam3" in sys.argv
    vid, vid_nav, results = [], [], []
    for i in range(n_pick):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        prompt = PROMPT_TRAIN if i < n_pick - 2 else PROMPT_HELDOUT
        driver.reset()
        res = run_episode(r, driver, "pick", obj, prompt,
                          vid if i < vp else None)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    for i in range(n_nav):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        driver.reset()
        nv = (vid if allcam3 else vid_nav) if i < vp else None
        res = run_episode(r, driver, "nav", obj, PROMPT_NAV, nv)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    for i in range(n_comp):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        driver.reset()
        res = run_episode(r, driver, "comp", obj, PROMPT_COMP,
                          vid if i < max(2, min(vp, 2)) else None)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    if vid:
        w = imageio.get_writer("/home/ucabhe0/Scratch/airvla/evalout/eval_preview.mp4",
                               fps=5, codec="libx264", quality=8,
                               macro_block_size=1)
        for fr in vid:
            w.append_data(fr)
        w.close()
    if vid_nav:
        w = imageio.get_writer("/home/ucabhe0/Scratch/airvla/evalout/eval_preview_nav.mp4",
                               fps=5, codec="libx264", quality=8,
                               macro_block_size=1)
        for fr in vid_nav:
            w.append_data(fr)
        w.close()
    ok = sum(1 for x in results if x["success"])
    print("EVAL-DONE mode=" + mode + " " + str(ok) + "/"
          + str(len(results)) + " success", flush=True)


if __name__ == "__main__":
    main()
