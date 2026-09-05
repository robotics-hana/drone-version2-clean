"""E3 fine-tune launcher: builds the lerobot-train argv from the
merged split (the episode list is too large to inline in a job file)
and runs the stock trainer. Fine-tunes FROM checkpoint 047500 (the
selected v2 policy) on the combined airvla_v21 train split, with the
cosine schedule compressed to the run length (decay 15000 = steps) so
the fine-tune completes a full warmup->decay cycle instead of riding
30k-schedule fragments. Everything else matches the frozen v2 recipe
(batch 4, seed 1000, bf16, grad ckpt, vision encoder unfrozen)."""
import json
import runpy
import sys

split = json.load(open("/home/ucabhe0/Scratch/airvla/sim/e3_split.json"))
train = split["train"]
print("E3TRAIN episodes:", len(train), flush=True)
sys.argv = [
    "lerobot-train",
    "--dataset.repo_id=hanapasta/airvla_v21",
    "--dataset.episodes=%s" % json.dumps(train, separators=(",", ":")),
    "--policy.path=/home/ucabhe0/Scratch/airvla/pi0_v2_out/checkpoints/047500/pretrained_model",
    '--rename_map={"observation.images.camera3": "observation.images.base_0_rgb", '
    '"observation.images.camera1": "observation.images.left_wrist_0_rgb", '
    '"observation.images.camera2": "observation.images.right_wrist_0_rgb"}',
    "--output_dir=/home/ucabhe0/Scratch/airvla/pi0_e3_out",
    "--batch_size=4",
    "--steps=15000",
    "--save_freq=2500",
    "--num_workers=0",
    "--seed=1000",
    "--policy.device=cuda",
    "--policy.dtype=bfloat16",
    "--policy.gradient_checkpointing=true",
    "--policy.freeze_vision_encoder=false",
    "--policy.train_expert_only=false",
    "--policy.scheduler_decay_steps=15000",
    "--wandb.enable=false",
    "--policy.push_to_hub=false",
]
runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
