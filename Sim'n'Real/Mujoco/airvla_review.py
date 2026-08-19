"""3-camera annotated review video for the AirVLA sample dataset:
wrist | forward | external, command + episode banner.

Tiles are upscaled 2x (256 -> 512) FOR VIEWING ONLY -- the dataset stays at
the paper's 256x256 policy resolution (itself now 2x-supersampled at render
time, so the upscale has clean anti-aliased pixels to work with)."""
import pathlib
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = (pathlib.Path.home() / ".cache/huggingface/lerobot/hanapasta"
        / "airvla_sample15")
OUT = pathlib.Path.home() / "airvla_review15.mp4"
CAMS = ["camera1", "camera2", "camera3"]
LABELS = ["wrist/down", "forward", "external"]
T = 512                      # displayed tile size (2x the 256 px dataset)

data = pd.read_parquet(ROOT / "data" / "chunk-000" / "file-000.parquet",
                       columns=["episode_index", "task_index"])
tasks = pd.read_parquet(ROOT / "meta" / "tasks.parquet")
if "task" in tasks.columns:
    tmap = dict(zip(tasks.get("task_index", range(len(tasks))), tasks["task"]))
else:
    tmap = {int(v): str(k) for k, v in
            zip(tasks.index, tasks[tasks.columns[0]])}
font = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
font_s = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
readers = [imageio.get_reader(ROOT / "videos" / f"observation.images.{c}"
                              / "chunk-000" / "file-000.mp4") for c in CAMS]
writer = imageio.get_writer(OUT, fps=10, codec="libx264", quality=8,
                            macro_block_size=1)
eps = data["episode_index"].to_numpy()
tis = data["task_index"].to_numpy()
n = 0
for i, frames in enumerate(zip(*readers)):
    if i >= len(eps):
        break
    big = [Image.fromarray(f).resize((T, T), Image.LANCZOS) for f in frames]
    strip = Image.new("RGB", (3 * T, T))
    for k, im in enumerate(big):
        strip.paste(im, (k * T, 0))
    dr = ImageDraw.Draw(strip, "RGBA")
    for k, lab in enumerate(LABELS):
        w = dr.textlength(lab, font=font_s)
        dr.rectangle([k*T+8, T-36, k*T+8+w+14, T-8], fill=(0, 0, 0, 160))
        dr.text((k*T+15, T-33), lab, font=font_s, fill=(180, 220, 255))
    label = tmap.get(int(tis[i]), "?")
    head = f"episode {int(eps[i])+1:02d}/15"
    w = max(dr.textlength(label, font=font), dr.textlength(head, font=font_s))
    dr.rectangle([8, 8, 8+w+20, 80], fill=(0, 0, 0, 180))
    dr.text((18, 12), head, font=font_s, fill=(255, 210, 80))
    dr.text((18, 40), label, font=font, fill=(255, 255, 255))
    writer.append_data(np.asarray(strip))
    n += 1
writer.close()
print(f"wrote {OUT} ({n} frames)")
