"""Plan D fine-tune launcher: lerobot-train argv from the merged v22
split. Fine-tunes FROM C-15000 (the selected Plan C policy -- D
builds on C, per the composition logic Hana set with H1C) on
airvla_v22 with PLAN C's EXACT optimizer recipe -- peak LR 5e-6,
cosine to 5e-7 over 20,000 steps, full warmup->decay cycle (A
churned at 2.5e-5; B under-learned; C passed its gates on this
schedule; per Hana 2026-09-08 the LR is deliberately NOT retuned so
any grounding movement is attributable to the DATA alone).

Episode list: C's own training list REBUILT VERBATIM (same
deterministic rng, seed 424244, over the same e3 manifest/split --
preserving C's internal old/new balance so its behaviour is not
forgotten) + ALL Plan-D train pairs (combined_episode_index >= 910).
Ratio lands near 45/55 old/new, the neighbourhood C validated.
"""
import json
import runpy
import sys

import numpy as np

SIM = "/home/ucabhe0/Scratch/airvla/sim"

# --- rebuild Plan C's list verbatim (its wrapper's exact logic) ---
e3_split = json.load(open(SIM + "/e3_split.json"))
e3_man = [json.loads(l) for l in
          open(SIM + "/e3_manifest_combined.jsonl")]
kind = {r["combined_episode_index"]: r["kind"] for r in e3_man}
new_tr = [i for i in e3_split["train"]
          if kind[i] in ("term", "pairA", "pairB")]
old_tr = [i for i in e3_split["train"]
          if kind[i] in ("std", "corr", "nav")]
rng = np.random.default_rng(424244)
keep = []
for k in ("std", "corr", "nav"):
    grp = [i for i in old_tr if kind[i] == k]
    rng.shuffle(grp)
    keep += grp[: len(grp) // 2]
c_list = sorted(new_tr + keep)

# --- add ALL Plan-D train pairs (v22 indices 910+) ---
d_split = json.load(open(SIM + "/d_split.json"))
d_tr = [i for i in d_split["train"] if i >= 910]
train = sorted(c_list + d_tr)
print("PLAND list: %d C-list + %d D-pairs = %d"
      % (len(c_list), len(d_tr), len(train)), flush=True)

sys.argv = [
    "lerobot-train",
    "--dataset.repo_id=hanapasta/airvla_v22",
    "--dataset.episodes=%s" % json.dumps(train, separators=(",", ":")),
    "--policy.path=/home/ucabhe0/Scratch/airvla/pi0_e3c_out/checkpoints/015000/pretrained_model",
    '--rename_map={"observation.images.camera3": "observation.images.base_0_rgb", '
    '"observation.images.camera1": "observation.images.left_wrist_0_rgb", '
    '"observation.images.camera2": "observation.images.right_wrist_0_rgb"}',
    "--output_dir=/home/ucabhe0/Scratch/airvla/pi0_d_out",
    "--batch_size=4",
    "--steps=20000",
    "--save_freq=2500",
    "--num_workers=0",
    "--seed=1000",
    "--policy.device=cuda",
    "--policy.dtype=bfloat16",
    "--policy.gradient_checkpointing=true",
    "--policy.freeze_vision_encoder=false",
    "--policy.train_expert_only=false",
    "--policy.scheduler_decay_steps=20000",
    "--policy.optimizer_lr=5e-6",
    "--policy.scheduler_decay_lr=5e-7",
    "--wandb.enable=false",
    "--policy.push_to_hub=false",
]
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
