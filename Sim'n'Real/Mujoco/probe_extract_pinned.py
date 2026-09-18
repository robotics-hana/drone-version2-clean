"""Re-extract decision-frame activations with the flow-matching noise PINNED,
and measure how much of each stack's activation is noise rather than stimulus.

    python probe_extract_pinned.py --checkpoint CKPT --dataset REPO \
        --root DROOT --meta episodes_meta.json \
        --out decision_activations_pinned.npz --audit activation_noise_audit.json

WHY THIS FILE EXISTS (it does NOT replace probe_extract_pi0.py)
----------------------------------------------------------------
probe_extract_pi0.py captures the residual stream during a call to
predict_action_chunk. For the TRUNK that is harmless: the PaliGemma prefix
sees images and language only, so its activations are a deterministic function
of the stimulus.

The EXPERT is different. pi0 is a flow-matching policy, and the expert stack
consumes the noisy action tokens being denoised. Those tokens start from

    modeling_pi0.sample_noise()   torch.normal(0., 1., size=shape)   # unseeded

so the expert's residual stream is a function of the stimulus AND of an
unseeded random draw. probe_extract_pi0.py took one draw per episode. Any
noise that leaks into the captured activation is variance that a probe cannot
explain, which depresses AUROC -- so a weak expert-side result is ambiguous
between "the model does not encode this" and "the encoding was buried under
sampling noise".

That ambiguity matters for exactly one headline number: the expert separates
refuse_hazard from act at CV AUROC 0.551 (null 0.496), i.e. at chance. Before
that can be reported as a property of the checkpoint, it has to be shown not
to be a property of the extraction.

WHAT IT MEASURES
----------------
1. AUDIT. For a subset of episodes, the same stimulus is pushed through the
   policy REPEATS times with independent noise draws. That gives, per stack
   and per layer:

     within   mean variance across noise draws of the SAME episode  (noise)
     between  variance across episode means                         (signal)
     noise_fraction = within / (within + between)

   The trunk is the built-in negative control: it cannot depend on the action
   noise, so its noise_fraction must come out at ~0. If it does not, the
   audit itself is broken and nothing downstream is trustworthy. The expert's
   number is the one being measured.

2. RE-EXTRACTION. Every episode is then re-extracted once with a pinned,
   per-episode noise tensor, so the npz is a deterministic function of the
   stimulus. Re-running probe_cv.py on it answers the question directly: if
   the expert refusal AUROC is unchanged, the null is real; if it rises, the
   original null was an extraction artefact.

3. EMITTED CHUNKS. The action chunk returned by each pinned forward pass is
   stored alongside the activations, at no extra GPU cost. behaviour_check.py
   measured how well the emitted chunk separates act from refuse (CV AUROC
   0.588) using unpinned noise, so that number carried the same contamination.
   Recomputing it from these chunks removes it.

Capture points and layer indexing are identical to probe_extract_pi0.py: the
INPUT to each block's input_layernorm, which IS the residual stream entering
the block, plus the input to the stack's final norm. That gives NL+1 entries
per stack, of which only the first NL are hookable layers.
"""
import argparse
import json
import pathlib

import numpy as np
import torch


def stacks_of(policy):
    pwe = policy.model.paligemma_with_expert
    return ({"trunk": pwe.paligemma.model.language_model.layers,
             "expert": pwe.gemma_expert.model.layers}, pwe)


def final_norms_of(pwe):
    return {"trunk": pwe.paligemma.model.language_model.norm,
            "expert": pwe.gemma_expert.model.norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="decision_activations_pinned.npz")
    ap.add_argument("--audit", default="activation_noise_audit.json")
    ap.add_argument("--n_var", type=int, default=12,
                    help="episodes used for the within-vs-between audit")
    ap.add_argument("--repeats", type=int, default=4,
                    help="independent noise draws per audit episode")
    ap.add_argument("--noise_seed", type=int, default=12345)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI0Policy.from_pretrained(args.checkpoint).to(dev).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": dev}})
    ds = (LeRobotDataset(args.dataset, root=args.root) if args.root
          else LeRobotDataset(args.dataset))
    payload = json.loads(pathlib.Path(args.meta).read_text())
    meta = payload["episodes"] if isinstance(payload, dict) else payload
    meta = [e for e in meta if e.get("decision_frame", -1) >= 0]
    n = len(meta)

    stacks, pwe = stacks_of(policy)
    norms = final_norms_of(pwe)
    nl = len(stacks["trunk"])
    width = {k: v[0].mlp.down_proj.out_features for k, v in stacks.items()}
    print(f"[paths] trunk {nl}L w{width['trunk']} | "
          f"expert {len(stacks['expert'])}L w{width['expert']} | {n} episodes",
          flush=True)

    cap, handles = {}, []

    def grab(key):
        def hook(_m, a):
            cap[key] = a[0].detach()
        return hook

    for name, layers in stacks.items():
        for i, layer in enumerate(layers):
            handles.append(
                layer.input_layernorm.register_forward_pre_hook(grab(f"{name}.{i}")))
        handles.append(norms[name].register_forward_pre_hook(grab(f"{name}.{nl}")))

    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    noise_map = {int(e["episode_index"]): torch.normal(
        0.0, 1.0, size=(1, chunk, adim), generator=gen).to(dev) for e in meta}

    def run_one(e, noise):
        """One forward pass. Returns {stack: (NL+1, width)} activations and the
        unnormalised action chunk."""
        erow = ds.meta.episodes[int(e["episode_index"])]
        gidx = int(erow["dataset_from_index"]) + int(e["decision_frame"])
        item = ds[gidx]
        raw = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
               for k, v in item.items()}
        cap.clear()
        kw = {"noise": noise} if noise is not None else {}
        with torch.no_grad():
            act = policy.predict_action_chunk(preprocessor(raw), **kw)
            # predict_action_chunk returns the RAW normalised model output;
            # unnormalising is a separate postprocessor step.
            act = postprocessor(act)
        got = {}
        for s in stacks:
            arr = np.zeros((nl + 1, width[s]), np.float32)
            for i in range(nl + 1):
                r = cap.get(f"{s}.{i}")
                if r is not None:
                    arr[i] = r[0].float().mean(0).cpu().numpy()
            got[s] = arr
        return got, act[0].float().cpu().numpy()

    # ---- 1. audit: within-episode (noise) vs between-episode (signal) -----
    sub = meta[:args.n_var]
    print(f"[audit] {len(sub)} episodes x {args.repeats} unpinned draws",
          flush=True)
    draws = {s: np.zeros((len(sub), args.repeats, nl + 1, width[s]), np.float32)
             for s in stacks}
    for i, e in enumerate(sub):
        for r in range(args.repeats):
            got, _ = run_one(e, None)          # None => pi0 draws its own noise
            for s in stacks:
                draws[s][i, r] = got[s]
        print(f"  audit {i+1}/{len(sub)}", flush=True)

    audit = {"n_episodes": len(sub), "repeats": args.repeats, "stacks": {}}
    for s in stacks:
        d = draws[s].astype(np.float64)
        # within: variance across repeats, averaged over episodes and units
        within = d.var(axis=1, ddof=1).mean(axis=(0, 2))            # (NL+1,)
        # between: variance across episode means
        between = d.mean(axis=1).var(axis=0, ddof=1).mean(axis=1)   # (NL+1,)
        frac = within / (within + between + 1e-30)
        audit["stacks"][s] = {
            "within_per_layer": within.tolist(),
            "between_per_layer": between.tolist(),
            "noise_fraction_per_layer": frac.tolist(),
            "noise_fraction_mean": float(frac.mean()),
            "noise_fraction_max": float(frac.max()),
        }
        print(f"[audit] {s:6s} noise_fraction mean {frac.mean():.4f} "
              f"max {frac.max():.4f}", flush=True)
    audit["interpretation"] = (
        "trunk is the negative control and must be ~0: it cannot see the "
        "action noise. The expert's value is the fraction of its activation "
        "variance that is sampling noise rather than stimulus.")
    pathlib.Path(args.audit).write_text(json.dumps(audit, indent=1))
    print(f"[save] {args.audit}", flush=True)

    # ---- 2. pinned re-extraction (same schema as probe_extract_pi0.py) ----
    out = {
        "episode_index": np.array([e["episode_index"] for e in meta], np.int32),
        "decision_frame": np.array([e["decision_frame"] for e in meta], np.int32),
        "mode_id": np.array([{"act": 0, "refuse_hazard": 1,
                              "refuse_ungrounded": 2}[e["mode"]] for e in meta],
                            np.int32),
        "is_pick": np.array([int(e["is_pick"]) for e in meta], np.int32),
    }
    for s in stacks:
        out[f"{s}_resid_mean"] = np.zeros((n, nl + 1, width[s]), np.float32)
    chunks = None
    for row, e in enumerate(meta):
        got, act = run_one(e, noise_map[int(e["episode_index"])])
        for s in stacks:
            out[f"{s}_resid_mean"][row] = got[s]
        if chunks is None:
            chunks = np.zeros((n,) + act.shape, np.float32)
        chunks[row] = act
        if (row + 1) % 20 == 0 or row + 1 == n:
            print(f"  extract {row+1}/{n}", flush=True)
    out["action_chunk_pinned"] = chunks

    for h in handles:
        h.remove()
    np.savez_compressed(args.out, **out)
    mb = pathlib.Path(args.out).stat().st_size / 1e6
    print(f"[save] {args.out} ({mb:.1f} MB) chunks{chunks.shape}")


if __name__ == "__main__":
    main()
