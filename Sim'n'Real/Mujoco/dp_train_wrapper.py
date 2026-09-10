"""Diffusion Policy baseline launcher (pre-registered 2026-09-10;
Sparks GB10, runs after ACT): from-scratch DP on airvla_v21, same
train split as Plan C, near-parity cadence (horizon 56 -- the
smallest UNet-compatible value >= 50+n_obs_steps-1 -- executing 50
of it, matching the naive-50 evaluation cadence).

usage: python dp_train_wrapper.py <out_dir> [--steps N] [--smoke]
  --smoke: 200 steps, save at 100 -- the mandatory
  does-the-code-run gate before the real training.
"""
import json
import runpy
import sys

OUT = sys.argv[1]
SMOKE = "--smoke" in sys.argv
STEPS = (int(sys.argv[sys.argv.index("--steps") + 1])
         if "--steps" in sys.argv else (200 if SMOKE else 50000))
SAVE = 100 if SMOKE else 5000

split = json.load(open("/clusterhome/hana/e3_split.json"))
train = split["train"]
print("DP train episodes:", len(train), flush=True)

sys.argv = [
    "lerobot-train",
    "--policy.type=diffusion",
    "--dataset.repo_id=hanapasta/airvla_v21",
    "--dataset.episodes=%s" % json.dumps(train, separators=(",", ":")),
    "--output_dir=%s" % OUT,
    "--batch_size=8",
    "--steps=%d" % STEPS,
    "--save_freq=%d" % SAVE,
    "--num_workers=4",
    "--seed=1000",
    "--policy.device=cuda",
    "--policy.horizon=56",
    "--policy.n_action_steps=50",
    "--wandb.enable=false",
    "--policy.push_to_hub=false",
]
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
