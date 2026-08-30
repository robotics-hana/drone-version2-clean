# Myriad cluster guide for Claude agents

Instructions for any Claude agent using UCL's Myriad cluster on this
account. The AirVLA dissertation experiments have priority through early
September 2026; your jobs queue AFTER them (§5). Written 2026-08-30.

## 1. Access and lay of the land

- Connect with `ssh myriad` (alias in `~/.ssh/config`; user `ucabhe0`).
- Scheduler is **SGE**, not Slurm: `qsub` / `qstat` / `qdel`, job scripts
  use `#$` directives. There is no `sbatch`, no `squeue`.
- OS is RHEL 7.9 with **glibc 2.17**. Any pip install must use
  `--only-binary=:all:` and only wheels tagged `manylinux_2_17` or older
  will load. cuDNN ≥ 9.8, torch > 2.6, and torchcodec are all known dead
  ends here — do not attempt them.
- GPU nodes are A100-PCIe-40GB (request with `-l gpu=1` and `-ac allow=L`).
  40 GB caps π₀-sized models at batch 4.
- `~/Scratch` is large but **not backed up**. Anything irreplaceable gets
  pushed off-cluster (HF, git) the day it is produced.
- `/tmp` is NOT shared between login nodes; `~` in remote-piped scripts
  expands locally — write absolute paths.

## 2. Hard rules (UCL policy — a violation notice has already been issued
##    once on this account; do not earn a second)

- **Login nodes: fewer than 6 cores and under 30 GB RAM, always.**
- **Never instantiate a model on a login node.** Not even "just to test
  loading". Model work happens inside batch jobs or `qrsh` sessions.
- Installs on the login node: serial (one pip at a time), `nice -n 10`,
  no parallel builds.
- Heavy CPU work (dataset processing, rendering, FK sweeps) goes in a
  batch job even when it would fit on the login node.
- Light monitoring (`qstat`, `tail` on logs, small greps) is fine.

## 3. Do not touch (AirVLA-owned, same account)

- `~/Scratch/airvla/**` — job scripts, logs, eval outputs, checkpoints.
- `~/Scratch/hf_cache/lerobot/hanapasta/**` — training datasets
  (`airvla_full`, `airvla_oracle`).
- `/home/ucabhe0/Scratch/env-pin/**` — the pinned Python environment.
  **Never install, upgrade, or remove anything in it.** It contains a
  hand-patched PyAV path (`PYAV14-SEEK-FALLBACK` in lerobot's
  `video_utils.py`) and exact pins that took a full night to converge.
  If you need Python packages, build your own venv elsewhere in Scratch
  with the same `--only-binary=:all:` discipline.
- `~/.cache/huggingface/token` — never read, print, or move it.
- **Never `qdel`, `qhold`, or `qalter` any job whose name starts with
  `airvla`** — they share this account, so the scheduler will let you;
  do not.

## 4. Job script template (SGE)

```bash
#!/bin/bash -l
#$ -l gpu=1
#$ -ac allow=L
#$ -l h_rt=4:0:0
#$ -l mem=32G
#$ -N myproj_taskname
#$ -wd /home/ucabhe0/Scratch/<yourdir>/logs
#$ -o task.log
#$ -e task.err
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
# your commands, absolute paths, own venv
```

Non-negotiable hygiene, each learned the hard way here:

- **LF line endings only.** Windows-authored scripts fail on SGE with
  cryptic "invalid option" errors and local `bash -n` does not catch it.
  After writing remotely: `tr -d '\r' < job > tmp && mv tmp job`, then
  `bash -n job`, then verify the bytes AT THE DESTINATION.
- Anything using MuJoCo rendering needs `MUJOCO_GL=egl` **and a GPU
  job** — EGL fails on CPU-only nodes even for kinematics-only scripts
  that merely import a renderer.
- `qalter -hold_jid` is blocked by the site's JSV. If you need to change
  a dependency, delete and resubmit the dependent job.
- Guard expensive jobs: check inputs exist at the top and `exit 0`
  ("releasing GPU") rather than idling an allocation.
- Request realistic `h_rt`; 24 h+ workloads must be chained as
  resubmit-with-resume jobs.

## 5. Queueing behind the AirVLA work (required)

The AirVLA jobs (all named `airvla_*`) must run first. Submit every job
of yours held behind the entire current AirVLA set:

```bash
# 1. List the live AirVLA job ids:
AIRVLA=$(qstat | awk '$3 ~ /^airvla/ {print $1}' | paste -sd, -)
# 2. Submit held behind ALL of them (works for r, qw and hqw jobs):
qsub -hold_jid "$AIRVLA" yourjob.job
```

Re-derive `$AIRVLA` at each submission — the set changes as the pipeline
advances. If `qstat` shows no `airvla_*` jobs at all, submit normally.
Do not "borrow a slot while theirs are only queued": several AirVLA jobs
sit in `hqw` as dependency chains and must take GPUs the moment their
holds release.

## 6. Monitoring etiquette

- Poll with `qstat` no more than every few minutes; read progress from
  your job's own log files.
- If the cluster looks wedged, the answer is patience, not resubmission
  storms: queue priority accrues with wait time.
- Report outcomes from logs, not assumptions — a job leaving the queue
  is not evidence it succeeded; check its exit marker.
