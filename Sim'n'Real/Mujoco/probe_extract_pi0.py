"""Residual stream at the decision frame, for pi0. Writes decision_activations.npz.

    python probe_extract_pi0.py --checkpoint CKPT \
        --dataset hanapasta/pick_hold_v4s_train \
        --meta pick_hold_v4s_train_meta.json --out decision_activations.npz

WHY A SEPARATE FILE FROM probe_extract.py
-----------------------------------------
probe_extract.py is SmolVLA-specific in two ways that do not survive the move
to pi0, and both fail SILENTLY rather than raising:

 1. LAYER PATHS. SmolVLA:  vlm_with_expert.get_vlm_model().text_model.layers
                           vlm_with_expert.lm_expert.layers
    pi0:                   paligemma_with_expert.paligemma.model.language_model.layers
                           paligemma_with_expert.gemma_expert.model.layers

 2. RESIDUAL ARITHMETIC. probe_extract.py reconstructs the stream as
        resid_post = mlp(post_attention_layernorm(x)) + x
    captured with a pre-hook on post_attention_layernorm and a forward hook on
    mlp. pi0 does NOT do a plain add -- its block ends with
        resid = _gated_residual(after_first_residual, mlp_out, gate)
              = after_first_residual + mlp_out * gate
    where `gate` comes back from the layernorm (adaptive RMSNorm). For
    pi0_base the gate is None, so the plain-add reconstruction happens to be
    correct; for pi05 (use_adarms) it is NOT, and the SmolVLA recipe would
    silently produce a mis-scaled residual and therefore a wrong direction.

This file avoids the whole question by reading the residual stream where it is
unambiguous: the INPUT to each block's input_layernorm. That tensor IS the
residual stream entering the block, whatever the block does internally.

    resid_pre[L] == resid_post[L-1]

so layer indices are shifted by one relative to probe_extract.py. The final
entry is the input to the stack's final norm, i.e. the output of the last
block. Verified against lerobot/pi0_base: pre-hooks fire on 18/18 layers of
both stacks, the L0 capture equals the fed embeddings exactly, an identity
rewrite through the same hook is bit-exact, and a rank-1 projection moves the
output (relative 0.034).
"""
import argparse
import json
import pathlib

import numpy as np
import torch


def stacks_of(policy):
    pwe = policy.model.paligemma_with_expert
    return {"trunk": pwe.paligemma.model.language_model.layers,
            "expert": pwe.gemma_expert.model.layers}, pwe


def final_norms_of(pwe):
    return {"trunk": pwe.paligemma.model.language_model.norm,
            "expert": pwe.gemma_expert.model.norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--meta", required=True,
                    help="collector sidecar (episode_index -> decision_frame)")
    ap.add_argument("--out", default="decision_activations.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = LeRobotDataset(args.dataset)
    payload = json.loads(pathlib.Path(args.meta).read_text())
    meta = payload["episodes"] if isinstance(payload, dict) else payload
    meta = [e for e in meta if e.get("decision_frame", -1) >= 0]
    n = len(meta)

    stacks, pwe = stacks_of(policy)
    norms = final_norms_of(pwe)
    NL = len(stacks["trunk"])
    W = {k: v[0].mlp.down_proj.out_features for k, v in stacks.items()}
    print(f"[paths] trunk {NL}L w{W['trunk']} | expert {len(stacks['expert'])}L "
          f"w{W['expert']} | {n} episodes")

    cap, handles = {}, []

    def grab(key):
        def h(_m, a):
            cap[key] = a[0].detach()
        return h

    for name, layers in stacks.items():
        for i, L in enumerate(layers):
            handles.append(
                L.input_layernorm.register_forward_pre_hook(grab(f"{name}.{i}")))
        handles.append(
            norms[name].register_forward_pre_hook(grab(f"{name}.{NL}")))

    out = {
        "episode_index": np.array([e["episode_index"] for e in meta], np.int32),
        "decision_frame": np.array([e["decision_frame"] for e in meta], np.int32),
        "mode_id": np.array([{"act": 0, "refuse_hazard": 1,
                              "refuse_ungrounded": 2}[e["mode"]] for e in meta],
                            np.int32),
        "is_pick": np.array([int(e["is_pick"]) for e in meta], np.int32),
    }
    # NL+1 entries: the input to each block, plus the input to the final norm
    for s in stacks:
        out[f"{s}_resid_mean"] = np.zeros((n, NL + 1, W[s]), np.float32)
        out[f"{s}_resid_last"] = np.zeros((n, NL + 1, W[s]), np.float32)

    for row, e in enumerate(meta):
        erow = ds.meta.episodes[int(e["episode_index"])]
        g = int(erow["dataset_from_index"]) + int(e["decision_frame"])
        item = ds[g]
        raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
               for k, v in item.items()}
        cap.clear()
        with torch.no_grad():
            policy.predict_action_chunk(preprocessor(raw))
        for s in stacks:
            for i in range(NL + 1):
                r = cap.get(f"{s}.{i}")
                if r is None:
                    continue
                r = r[0].float()
                out[f"{s}_resid_mean"][row, i] = r.mean(0).cpu().numpy()
                out[f"{s}_resid_last"][row, i] = r[-1].cpu().numpy()
        if (row + 1) % 20 == 0 or row + 1 == n:
            print(f"  {row+1}/{n}", flush=True)

    for h in handles:
        h.remove()
    np.savez_compressed(args.out, **out)
    mb = pathlib.Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out} ({mb:.1f} MB) "
          f"trunk{out['trunk_resid_mean'].shape} expert{out['expert_resid_mean'].shape}")


if __name__ == "__main__":
    main()
