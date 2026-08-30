"""Copy the collector's sidecar into the LeRobot dataset root.

    python install_sidecar.py pick_hold_v4s_train_meta.json hanapasta/pick_hold_v4s_train

probe_extract.py and ablate_refusal.py both read the per-episode metadata from
<dataset root>/episodes_meta.json. collect_v4style.py writes it to the working
directory under its own name, because the dataset root does not exist until
LeRobotDataset.create runs and the file has to survive a crashed collection.
This puts it where the existing tooling looks, without either side having to
know about the other.
"""
import json
import pathlib
import shutil
import sys

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
src, repo_id = pathlib.Path(sys.argv[1]), sys.argv[2]
from lerobot.utils.constants import HF_LEROBOT_HOME
root = pathlib.Path(HF_LEROBOT_HOME) / repo_id
if not root.exists():
    raise SystemExit(f"no dataset at {root}")
payload = json.loads(src.read_text())
eps = payload["episodes"] if isinstance(payload, dict) else payload
dst = root / "episodes_meta.json"
dst.write_text(json.dumps({"episodes": eps}, indent=1))
print(f"installed {len(eps)} episode records -> {dst}")
