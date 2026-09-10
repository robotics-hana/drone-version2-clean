"""ACT baseline launcher (pre-registered 2026-09-10; runs on Sparks
GB10 in parallel with Myriad's Plan D): from-scratch ACT on
airvla_v21 with the SAME train split as Plan C (e3_split train,
728 episodes) and naive-50 cadence parity (chunk 50, execute 50).
The point of the rung: was a 4B pretrained VLA necessary, or does a
small from-scratch imitation policy reach the same place?

usage: python act_train_wrapper.py <out_dir> [--steps N] [--smoke]
  --smoke: 200 steps, save at 100 -- the mandatory
  does-the-code-run gate before the real training (Hana's
  mini-first directive).
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
print("ACT train episodes:", len(train), flush=True)

sys.argv = [
    "lerobot-train",
    "--policy.type=act",
    "--dataset.repo_id=hanapasta/airvla_v21",
    "--dataset.episodes=%s" % json.dumps(train, separators=(",", ":")),
    "--output_dir=%s" % OUT,
    "--batch_size=8",
    "--steps=%d" % STEPS,
    "--save_freq=%d" % SAVE,
    "--num_workers=4",
    "--seed=1000",
    "--policy.device=cuda",
    "--policy.chunk_size=50",
    "--policy.n_action_steps=50",
    "--wandb.enable=false",
    "--policy.push_to_hub=false",
]
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
