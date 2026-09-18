"""What does ablating the refusal direction do to ACT episodes, and in which
direction?

    python ablate_on_act.py --direction refusal --layers 16,17 ...

WHY THIS FILE EXISTS
--------------------
The paired layer sweep already establishes that removing the refusal direction
from act episodes changes them a lot: relative displacement 0.298 at expert
L16, against 0.004 for a norm-matched random direction — about seventy times
the control. But that statistic is a NORM. It says the act behaviour moved; it
cannot say whether it moved toward the demonstrated act or away from it, and
those imply opposite things.

The hypothesis this is built to test comes from two earlier results read
together:

  * On act episodes the policy commands joint2 = +0.099 where the
    demonstrations command +0.373 — it under-extends the arm to about a
    quarter of what the task needs, and the closed-loop rollout misses the
    block laterally by 55 mm with the jaws shut.
  * The refusal axis is causally sufficient: injecting it at L17 drives joint2
    to -0.51, most of the way to the demonstrated abstain pose.

If that axis is partially ACTIVE during act episodes — leaking, rather than
being cleanly gated off when the scene contains no hazard — then it is pulling
joint2 toward the abstain pose on episodes where the policy should be
reaching. That would make the arm under-extension a consequence of an
imperfectly gated refusal representation rather than a generic failure to fit
the channel.

    prediction if the axis leaks   ablating refusal on act episodes moves
        joint2 UP, toward the demonstrated +0.373
    prediction if it does not      joint2 barely moves, or moves down, and the
        0.298 displacement is spread over channels that do not carry the
        decision

Both outcomes are informative, and they are distinguished by the SIGN of a
single number, which is why this is worth a separate run rather than an
inference from the norm.

CONTROLS. The same ablation is applied for band (encoded but not the decision
variable) and for a norm-matched random direction. If joint2 rises for every
direction, the effect is generic and the refusal reading is wrong.

Noise is pinned per episode, so baseline and ablated chunks differ only by the
projection — see noise_paired_ablation.py for why that is not optional.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
from ablate_sweeps import deflated_directions  # noqa: E402

CHANNELS = ["x", "y", "z", "joint1", "joint2", "gripper"]
JOINT2, GRIPPER = 4, 5

SEL = {
    "refusal": (lambda r: r["mode"] == "refuse_hazard",
                lambda r: r["mode"] == "act"),
    "ungrounded": (lambda r: r["mode"] == "refuse_ungrounded",
                   lambda r: r["mode"] == "act"),
    "band": (lambda r: r["mode"] == "act" and r.get("band") == "far",
             lambda r: r["mode"] == "act" and r.get("band") == "near"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="refusal",
                    choices=list(SEL) + ["random"])
    ap.add_argument("--stack", default="expert", choices=["trunk", "expert"])
    ap.add_argument("--layers", default="16,17",
                    help="comma-separated; ablated together")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = (LeRobotDataset(args.dataset, root=args.root) if args.root
          else LeRobotDataset(args.dataset))
    payload = json.loads(pathlib.Path(args.meta).read_text())
    eps = payload["episodes"] if isinstance(payload, dict) else payload

    pwe = policy.model.paligemma_with_expert
    stacks = {"trunk": pwe.paligemma.model.language_model.layers,
              "expert": pwe.gemma_expert.model.layers}
    n_hookable = len(stacks[args.stack])
    layers = [int(x) for x in args.layers.split(",") if int(x) < n_hookable]

    key = f"{args.stack}_resid_mean"
    if args.direction == "random":
        z = np.load(args.npz)
        arr = z[key]
        rng = np.random.default_rng(0)
        vecs = np.zeros((arr.shape[1], 1, arr.shape[2]), np.float32)
        for layer in range(arr.shape[1]):
            v = rng.normal(size=arr.shape[2])
            vecs[layer, 0] = (v / np.linalg.norm(v)).astype(np.float32)
    else:
        vecs = deflated_directions(args.npz, args.meta, key,
                                   *SEL[args.direction], 1)

    act = [e for e in eps if e["mode"] == "act"
           and e.get("decision_frame", -1) >= 0][:args.n_episodes]
    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise_map = {int(e["episode_index"]): torch.normal(
        0.0, 1.0, size=(1, chunk, adim), generator=gen).to(dev) for e in act}

    handles = []

    def rewrite(vk):
        vt = torch.tensor(vk, dtype=torch.float32)

        def hook(_m, a):
            x = a[0]
            vd = vt.to(x.device, x.dtype)
            return (x - (x @ vd.T) @ vd,) + tuple(a[1:])
        return hook

    def install():
        for layer in layers:
            handles.append(stacks[args.stack][layer].input_layernorm
                           .register_forward_pre_hook(rewrite(vecs[layer, :1])))

    def clear():
        while handles:
            handles.pop().remove()

    def emit(rows):
        out = []
        for e in rows:
            erow = ds.meta.episodes[int(e["episode_index"])]
            g = int(erow["dataset_from_index"]) + int(e["decision_frame"])
            item = ds[g]
            raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                   for k, v in item.items()}
            with torch.no_grad():
                a = policy.predict_action_chunk(
                    pre(raw), noise=noise_map[int(e["episode_index"])])
                a = post(a)          # real units
            out.append(a[0].float().cpu().numpy())
        return np.stack(out)

    base = emit(act)
    install()
    try:
        abl = emit(act)
    finally:
        clear()

    per = {}
    for i, nm in enumerate(CHANNELS):
        b, a_ = float(base[..., i].mean()), float(abl[..., i].mean())
        per[nm] = {"baseline": b, "ablated": a_, "delta": a_ - b}
    displacement = float(np.linalg.norm(abl - base, axis=(1, 2)).mean()
                         / (np.linalg.norm(base, axis=(1, 2)).mean() + 1e-9))

    out = {"direction": args.direction, "stack": args.stack, "layers": layers,
           "n_act": len(act), "paired_noise": True,
           "displacement": displacement, "per_channel": per,
           "demo_act_joint2": 0.3733, "demo_refuse_joint2": -0.4629}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"[{args.direction} @ {args.stack} L{layers}] n={len(act)} act episodes")
    print(f"  overall displacement {displacement:.4f}")
    print(f"  {'channel':9s} {'baseline':>10s} {'ablated':>10s} {'delta':>10s}")
    for nm in CHANNELS:
        p = per[nm]
        mark = "  <-- decision channel" if nm == "joint2" else ""
        print(f"  {nm:9s} {p['baseline']:10.4f} {p['ablated']:10.4f} "
              f"{p['delta']:+10.4f}{mark}")
    j = per["joint2"]
    print(f"  demonstrated act joint2 = +0.3733; ablation moved joint2 "
          f"{'TOWARD' if j['delta'] > 0 else 'AWAY FROM'} it "
          f"by {abs(j['delta']):.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
