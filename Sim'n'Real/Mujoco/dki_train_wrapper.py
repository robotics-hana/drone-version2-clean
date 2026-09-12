"""D-KI launcher (pre-registered 2026-09-12, Hana's approval): the
Knowledge-Insulation rerun of Plan D. IDENTICAL to d_train_wrapper
in data (airvla_v22), episode list (C-list rebuilt verbatim + all D
train pairs), steps, schedule and seed — with exactly one
substantive change, per Driess et al. 2025 (arXiv:2505.23705, the
pi0 authors): the VLM backbone and vision encoder are FROZEN and
only the action expert trains, so action gradients cannot erode the
pretrained language representations.

Hypothesis: Plan D's grounding regression (flew-to-target 40/60 vs
C's 47/60 despite better val MSE) is backbone erosion under full
fine-tuning. If D-KI recovers flew-to-target >= 47/60, erosion is
confirmed and scaled pairs + insulation become the positive result;
if not, the overdose finding survives its strongest objection.

usage: python dki_train_wrapper.py [--smoke]
  --smoke: 200 steps to a scratch dir -- the mandatory
  does-the-code-run gate before the real training.
"""
import json
import runpy
import sys

SIM = "/home/ucabhe0/Scratch/airvla/sim"
SMOKE = "--smoke" in sys.argv
OUT = ("/home/ucabhe0/Scratch/airvla/pi0_dki_smoke" if SMOKE
       else "/home/ucabhe0/Scratch/airvla/pi0_dki_out")
STEPS = 200 if SMOKE else 20000
SAVE = 100 if SMOKE else 2500

import numpy as np

# --- rebuild Plan C's list verbatim (same as d_train_wrapper) ---
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

d_split = json.load(open(SIM + "/d_split.json"))
d_tr = [i for i in d_split["train"] if i >= 910]
train = sorted(c_list + d_tr)
print("DKI list: %d C-list + %d D-pairs = %d (smoke=%s)"
      % (len(c_list), len(d_tr), len(train), SMOKE), flush=True)

sys.argv = [
    "lerobot-train",
    "--dataset.repo_id=hanapasta/airvla_v22",
    "--dataset.episodes=%s" % json.dumps(train, separators=(",", ":")),
    "--policy.path=/home/ucabhe0/Scratch/airvla/pi0_e3c_out/checkpoints/015000/pretrained_model",
    '--rename_map={"observation.images.camera3": "observation.images.base_0_rgb", '
    '"observation.images.camera1": "observation.images.left_wrist_0_rgb", '
    '"observation.images.camera2": "observation.images.right_wrist_0_rgb"}',
    "--output_dir=%s" % OUT,
    "--batch_size=4",
    "--steps=%d" % STEPS,
    "--save_freq=%d" % SAVE,
    "--num_workers=0",
    "--seed=1000",
    "--policy.device=cuda",
    "--policy.dtype=bfloat16",
    "--policy.gradient_checkpointing=true",
    # THE ONE SUBSTANTIVE CHANGE vs d_train_wrapper (KI recipe):
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    "--policy.scheduler_decay_steps=20000",
    "--policy.optimizer_lr=5e-6",
    "--policy.scheduler_decay_lr=5e-7",
    "--wandb.enable=false",
    "--policy.push_to_hub=false",
]
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
