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
TRAJ_PATH = "/clusterhome/hana/eval_traj.jsonl"   # per-episode drone/
MODE_TAG = ["?"]      # object trajectories for the results dashboard


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

    def tick(self, act):
        r, ex = self.r, self.ex
        self.sp = self.sp + np.clip(act[0:3], -0.03, 0.03)
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
    f["observation.state"] = torch.from_numpy(
        r.state()).unsqueeze(0).to(dev).to(dtype)
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
    plat = Platform(r, start, yaw0)
    adr = r.cur["adr"]
    lifted = crossed = hover_done = wrong_order = False
    hover_run = 0
    miss = 1e9        # closest jaws-to-grasp-target approach (near-miss
    y_prev = float(r.data.qpos[1])   # metric: analyzable even at 0%)
    traj = []
    t = 0
    for t in range(limit):
        if driver is None:                      # stub: drift forward
            act = np.array([0, 0.002, 0, 0, 0, 0, 1.0], np.float32)
        else:
            act = driver.act(r, task)
        plat.tick(act)
        if vid is not None and t % 2 == 0:
            if kind == "nav":
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
            traj.append([round(float(r.data.qpos[0]), 3),
                         round(float(r.data.qpos[1]), 3),
                         round(float(r.data.qpos[2]), 3)])
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
                mode=MODE_TAG[0], kind=kind, obj=obj,
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
                  and not wrong_order)
        return dict(kind=kind, obj=obj, prompt=prompt,
                    crossed=bool(crossed), hover=bool(hover_done),
                    picked=bool(lifted), placed=placed,
                    wrong_order=bool(wrong_order), success=ok,
                    miss_mm=round(miss * 1000, 1), frames=int(t + 1))
    return dict(kind=kind, obj=obj, prompt=prompt, crossed=bool(crossed),
                success=bool(hover_done), frames=int(t + 1))


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
    pos = [a for a in sys.argv[2:] if a.isdigit()]
    n_pick = int(pos[0]) if len(pos) > 0 else 6
    n_nav = int(pos[1]) if len(pos) > 1 else 3
    n_comp = int(pos[2]) if len(pos) > 2 else 0
    dev = "cpu" if "cpu" in sys.argv else "cuda"
    dtype = torch.float32

    policy = PI0Policy.from_pretrained(ckpt)
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

    vid, vid_nav, results = [], [], []
    for i in range(n_pick):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        prompt = PROMPT_TRAIN if i < n_pick - 2 else PROMPT_HELDOUT
        driver.reset()
        res = run_episode(r, driver, "pick", obj, prompt,
                          vid if i < VID_EPS else None)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    for i in range(n_nav):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        driver.reset()
        res = run_episode(r, driver, "nav", obj, PROMPT_NAV,
                          vid_nav if i < VID_EPS else None)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    for i in range(n_comp):
        obj = "plush penguin" if i % 2 == 0 else "weight"
        driver.reset()
        res = run_episode(r, driver, "comp", obj, PROMPT_COMP,
                          vid if i < 1 else None)
        results.append(res)
        print("EVAL", json.dumps(res), flush=True)
    if vid:
        w = imageio.get_writer("/clusterhome/hana/eval_preview.mp4",
                               fps=5, codec="libx264", quality=8,
                               macro_block_size=1)
        for fr in vid:
            w.append_data(fr)
        w.close()
    if vid_nav:
        w = imageio.get_writer("/clusterhome/hana/eval_preview_nav.mp4",
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
