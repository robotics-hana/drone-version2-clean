# π0 frozen-trunk training — complete pipeline

## One command

```bash
cd /home/ec2-user/SageMaker/skygrip
python -c "from huggingface_hub import snapshot_download as d; d('hanapasta/skygrip-sagemaker',repo_type='dataset',local_dir='.')"
bash sm_all.sh s1
```

`sm_all.sh` chains setup → auth check → pull → launch (or resume) → reaper.
It is **idempotent**: re-run it after any instance restart; finished steps are
skipped. Then paste `sm_monitor.py` into a notebook cell with `TAG = "s1"`.

## Files

| file | where | what |
|---|---|---|
| `sm_all.sh` | terminal | **everything, one command** |
| `sm_setup.sh` | terminal | instance report + deps + auto-tuned `sm_config.env` |
| `sm_pull.py` | terminal | pi0_base (14 GB) + dataset + sidecar, verify features |
| `sm_train.sh` | terminal | launch detached (tmux, else setsid+nohup) |
| `sm_reap.sh` | terminal (bg) | keep only newest N checkpoints |
| `sm_monitor.py` | **notebook cell** | tail log, flag decode-bound, keep session alive |
| `sm_resume.sh` | terminal | resume from last checkpoint |
| `sm_export.py` | terminal | push finished checkpoint to the Hub |
| `colab_train_pi0.ipynb` | Colab | same run on an A100 (second replicate) |
| `push_from_sparks.py` | sparks | publish datasets — **already done** |
| `probe_extract_pi0.py` | after training | residual stream at the decision frame → `.npz` |
| `direction_controls.py` | after training | refusal + behaviour-matched control directions |
| `ablate_sweeps.py` | after training | layer sweep and rank-k sweep |
| `install_sidecar.py` | either | put `episodes_meta.json` where the tools look |

## Auto-tuning

`sm_setup.sh` inspects the box and writes `sm_config.env`, which `sm_train.sh`
reads:

| VRAM | batch | grad ckpt |
|---|---|---|
| ≥ 38 GB (A100) | 16 | off |
| ≥ 22 GB (A10G 24 GB) | 8 | off |
| ≥ 15 GB (T4) | 4 | on |
| below | 2 | on |

`num_workers` = vCPUs − 1, and it states plainly whether you will be
**decode-bound**: the dataset is video-backed (298k frames × 3 streams decoded
on the fly), so ≤ 4 vCPUs starves the GPU and the run takes far longer than the
GPU alone implies. That is the number that decides SageMaker vs Colab.

## The three failure modes, and what handles each

1. **Instance restart wipes `$HOME`.** Only `/home/ec2-user/SageMaker`
   persists. Every script resolves that as `$ROOT` — venv, HF cache,
   checkpoints and logs all live there. Re-run `sm_all.sh` after a restart.
2. **Disk fills mid-run.** A π0 checkpoint is 6–9 GB and lerobot never prunes.
   `sm_reap.sh` keeps the newest two; `sm_all.sh` starts it automatically when
   free space is under 40 GB. Raising the EBS volume to ≥100 GB is better.
3. **Decode-bound GPU.** `sm_monitor.py` warns below 70% utilisation. Raise
   `NUM_WORKERS` in `sm_config.env` and relaunch.

## Disconnection

Training runs detached, so closing the browser or losing internet does not
touch it; the notebook cell is only a viewer. Only the *instance* stopping
kills it — hence `--save_freq=2000` plus `sm_resume.sh` (and `sm_all.sh`
auto-resumes if checkpoints already exist).

Pinned to the versions the dataset was built with: python 3.12, torch 2.11,
**lerobot 0.6.0**, dataset `codebase_version v3.0`.
