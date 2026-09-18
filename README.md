# SkyGrip

**Whole-Body Control for Quadcopter + Manipulator**  
A unified dynamics-based framework for **simulation–real robot synchronization**, supporting MuJoCo physics simulation, URDF/MJCF modeling, (optional) PyTorch for optimization/learning, and direct control of real Dynamixel servos with position, PWM, or torque commands.

> ✅ Highlights:  
> - **MuJoCo** real-time simulation (direct URDF load, or MJCF with actuators)  
> - **Real–Sim sync**: read joint state from hardware, inject into simulation; send sim torques → real PWM  
> - **Complete control modes**: position, PWM, torque (Nm → PWM approx.)  
> - **3D modeling pipeline**: SolidWorks → URDF (STL meshes) → MJCF  
> - **(Optional) PyTorch**: for MPPI/MPC, neural policies, residual models

---

## Project Structure

```

.
├─ 3D Model/           # CAD assets (SolidWorks / STL)
├─ Mujoco/             # MuJoCo simulation scripts & model demos (e.g. test1.py)
├─ PyBullet/           # PyBullet simulation scripts
├─ SkyGrip_URDF/       # Manipulator URDF and STL meshes
└─ Reports/            # Notes, reports, parameter tuning logs

````

---

## Environment & Dependencies

### 1) Python & Conda
```bash
conda create -n wbc python=3.10 -y
conda activate wbc
````

### 2) Core packages

```bash
pip install mujoco numpy scipy
# Optional: learning / optimization
pip install torch
# For Dynamixel hardware
pip install dynamixel-sdk
```

> MuJoCo ≥ 3.1 supports `pip install mujoco` directly — no need for the old `mujoco-py`.

---

## Quick Start (MuJoCo)

### A. Direct URDF load

```python
import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("URDF/mppi/mppi.urdf")
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
```

> URDF is fine for visualization, but MuJoCo **will not auto-create actuators** from `<transmission>` — add them manually in MJCF for control.

### B. MJCF with actuators (recommended for control)

1. Load URDF in MuJoCo Viewer → “Save XML” → `mppi.xml`
2. Add:

```xml
<actuator>
  <motor joint="Joint_1" ctrlrange="-1 1" gear="1"/>
  <motor joint="Joint_2" ctrlrange="-1 1" gear="1"/>
</actuator>
```

3. Run:

```python
model = mujoco.MjModel.from_xml_path("Mujoco/mppi.xml")
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        data.ctrl[:] = [0.2, -0.1]  # example torque commands
        mujoco.mj_step(model, data)
        v.sync()
```

---

## Dynamixel Control & Real–Sim Sync

### 1) Controller API (excerpt)

* **Position control**: `send_joint_positions([deg1, deg2, ...])` (auto-switch to Position mode)
* **PWM control**: `send_pwm([p1, p2, ...])` (auto-switch to PWM mode)
* **Torque control**: `send_torque([τ1, τ2, ...])` (Nm → PWM via `max_torque` & `max_pwm`)
* **Read state**: `get_joint_state() -> (qpos_rad[], qvel_rad_s[])`

⚠ PWM–torque mapping is **not perfectly linear**; use an approximate mapping first, then calibrate.

### 2) Sync real robot state into sim

```python
def apply_real_state_to_sim(model, data, qpos, qvel):
    data.qpos[:len(qpos)] = qpos
    data.qvel[:len(qvel)] = qvel
    mujoco.mj_forward(model, data)  # recompute derived quantities
```

Call before each sim step if doing frame-by-frame sync.

### 3) Synchronized torque controller

```python
class SynchronizedTorqueController:
    def __init__(self, real, model, data, enable_real=True, enable_sim=True):
        self.real, self.model, self.data = real, model, data
        self.enable_real, self.enable_sim = enable_real, enable_sim

    def send_torque(self, tau):
        if self.enable_sim:
            self.data.ctrl[:len(tau)] = tau
        if self.enable_real:
            self.real.send_torque(list(tau))
```

---

## 3D Modeling & Coordinates

* **SolidWorks → URDF**: with SW2URDF exporter (includes mass, inertia, STL meshes)
* **URDF → MJCF**: load URDF in MuJoCo Viewer, save as MJCF, then add actuators
* **Mesh orientation**: rotate `<body>` (with `quat=`) — not just `<geom>` — so children follow
* **Gravity/integrator**:

```xml
<option gravity="0 0 -9.81" timestep="0.001" integrator="RK4"/>
```

---

## Tuning & Debug Checklist

* **Jitter / instability**

  * Ensure `mass` ≥ 0.02 kg, `diaginertia` ≥ 1e-5
  * Add `damping` (0.05–0.2) and some `frictionloss`
  * Check for collision issues or `data.cfrc_ext` spikes
  * Zero out `data.ctrl[:]` if actuators retain old commands
* **Orientation mismatch**

  * Combine quaternions by multiplication, not addition
* **No actuator control from URDF**

  * MuJoCo ignores `<transmission>` for actuators; add `<actuator>` in MJCF
* **Sim–real gap**

  * Run a periodic reverse-torque experiment and compare qpos curves; adjust `damping`, `frictionloss`, mass, or torque mapping

---

## Optional PyTorch Integration

* Write your policy (MPPI/MPC/NN) in Torch, read sim state via `data.qpos/qvel`, output `tau` to `SynchronizedTorqueController`
* Keep policy logic separate from sim/env code

---

## Run Examples

* **MuJoCo demo**:

```bash
python Mujoco/test1.py
```

* **Sim–real consistency test**:

  * Apply fixed torque, flip sign every N frames
  * Plot difference between sim and real joint positions

---

## AirVLA: language-conditioned aerial manipulation (v2 campaign)

This repository contains the MSc-dissertation experiments that adapt a
pretrained **π₀ vision-language-action policy** to the SkyGrip drone +
2-DoF arm + parallel gripper, evaluated in MuJoCo. The policy receives
three camera images and a written instruction and outputs drone motion
and gripper commands; the arm follows deterministic task-phase logic.

Everything below is sufficient to reproduce the campaign end to end.
Per-experiment pre-registrations, amendments, results and provenance
are tracked in **[`Reports/finaldroneresults.md`](Reports/finaldroneresults.md)**,
which is the authoritative record — this README is the operational
guide.

**Headline result.** The pure VLA reliably selects and approaches the
commanded object but rarely grasps (1/60). Adding a specialised
terminal controller for the last 0.15 m raises this to **36/60 grasps
(60%) and 27/60 complete pick-and-place (45%)**.

---

### 1. Environment

The simulator version is **load-bearing**: training data and every
frozen evaluation were produced under **MuJoCo 3.3.4**. A different
MuJoCo changes contact behaviour enough to invalidate comparisons with
the banked results (an earlier v1 run had a hidden 3.11-data /
3.3.4-eval mismatch — see the ledger).

```bash
conda create -n airvla python=3.10 -y && conda activate airvla
pip install mujoco==3.3.4 numpy scipy torch imageio imageio-ffmpeg
pip install lerobot                 # π₀, ACT and Diffusion Policy implementations
pip install huggingface_hub
export MUJOCO_GL=egl                # headless rendering on cluster nodes
```

On UCL Myriad the campaign used a pinned interpreter at
`~/Scratch/env-pin/bin/python`; that pin was never modified mid-campaign
by design.

Working directory for all commands below: `Sim'n'Real/Mujoco/`.
The scene is `SkyGrip_airvla.xml`.

---

### 2. Where the data, weights and videos live

#### 2.1 Datasets (Hugging Face, LeRobot format)

| Repo | Contents | Episodes / frames | Visibility |
|---|---|---|---|
| `hanapasta/airvla_v2` | **v2 baseline**: 360 pick + 240 nav demos | 600 / 239,520 | private |
| `hanapasta/airvla_v21` | **v2 + F1** (terminal-corrective) — trains FT-C | 910 / 391,750 | public |
| `hanapasta/airvla_v22` | **v2 + F1 + F2** (paired-command scaling) — trains FT-D / FT-D-KI | 1358 / 622,801 | public |
| `hanapasta/airvla_v24` | **v21 + on-policy DAgger** — trains FT-DAG | 1110 / 512,577 | public |
| `hanapasta/airvla_v3` | v2 relabelled with arm-joint deltas — trains the arm ablation | 600 / 239,520 | private |
| `hanapasta/airvla_dag1` … `airvla_dag4` | raw DAgger collection shards (merged into v24) | — | public |
| `hanapasta/airvla_v2_d2` … `_d4` | Plan-D collection shards (merged into v22) | — | public |
| `hanapasta/airvla_v2_e3` | F1 corrective collection shard | — | public |

Download a dataset:

```bash
huggingface-cli download hanapasta/airvla_v21 --repo-type dataset \
  --local-dir ~/hf_cache/lerobot/hanapasta/airvla_v21
# If training offline, delete the legacy download marker or LeRobot
# forces a hub re-sync and fails under HF_HUB_OFFLINE=1:
rm -rf ~/hf_cache/lerobot/hanapasta/airvla_v21/.cache
```

#### 2.2 Trained weights

| Configuration | Hugging Face | Cluster path (Myriad) |
|---|---|---|
| **Base** (π₀, step 47500) | `hanapasta/airvla_v2_pi0_047500` (private) | `~/Scratch/airvla/pi0_v2_out/checkpoints/047500/pretrained_model` |
| **FT-C** (step 15000) | *not on hub* — see note | `~/Scratch/airvla/pi0_e3c_r_out/checkpoints/015000/pretrained_model` |
| **FT-D** (step 15000) | `hanapasta/airvla_d_15000` — **empty repo, push failed** | `~/Scratch/airvla/pi0_d_out/checkpoints/015000/pretrained_model` |
| **FT-D-KI** (step 5000) | *not on hub* | `~/Scratch/airvla/pi0_dki_out/checkpoints/005000/pretrained_model` |
| **FT-DAG** (step 17500) | *not on hub* | `~/Scratch/airvla/pi0_dag_out/checkpoints/017500/pretrained_model` |
| **V3-arm** (step 25000) | *not on hub* | `~/Scratch/airvla/pi0_v3_out/checkpoints/025000/pretrained_model` |
| **Learned terminal servo** | `hanapasta/airvla_e5_actor` (private) — actor + `rl_env.py` + `rl_nets.py` | `~/Scratch/airvla/e5_out/e5_actor_final.pt` |
| **ACT baseline** | `hanapasta/act_v21` (private) | `~/Scratch/airvla/act_ckpt/pretrained_model` |
| **Diffusion Policy baseline** | `hanapasta/dp_v21` (private) | `~/Scratch/airvla/dp_ckpt` |

> **Backup status (honest note).** Several π₀ checkpoints are
> cluster-only: hub backups are blocked by a Hugging Face private-repo
> storage limit, and `hanapasta/airvla_d_15000` currently contains only
> `.gitattributes` from a failed push. The FT-C checkpoint in the table
> is `pi0_e3c_r_out`, a deterministic **retrain** of the original
> C-15000, which was lost to a cleanup error and reproduced under
> equivalence gates (validation MSE 7.7×10⁻⁵ vs the original
> 7.679×10⁻⁵, plus a matching closed-loop mini). The full incident and
> recovery are documented in the ledger. Until the storage limit is
> resolved, **these weights exist in one place only.**

#### 2.3 Evaluation videos and logs

| Artefact | Location | Notes |
|---|---|---|
| Rendered evaluation episodes | `Reports/v2_eval_videos/` | 206 MP4s, ~4.6 GB — **git-ignored**, local only |
| Contact sheets / figures | `Reports/*.png` | tracked in git |
| Per-episode eval records | `~/Scratch/airvla/logs/*.log` (cluster) | one `EVAL {json}` line per episode |
| Trajectory dumps | `~/Scratch/airvla/sim/eval_v2_traj.jsonl` (cluster) | drone + jaw pose every 3 ticks |
| Collection manifests | `v2_manifest_<seed>.jsonl`, `dag_manifest_<seed>.jsonl` | one line per attempt, accepted or rejected |

Video sub-directories map to experiments: `full47k5` (Base full run, 80
clips), `h1mini`/`h1cmini`/`h1cfull` (scripted hybrid), `e5cfull`
(learned hybrid), `e1mini47k5`/`e2mini47k5`/`e2full47k5` (execution
horizon and RTC), `ooddfull` (novel-distractor OOD), `grasps` (isolated
successful grasps), plus 42 loose clips at the top level.

Videos are produced by passing `--video` to `eval_v2.py`, which writes
`v2vid_<tag>_pick<NN>.mp4` / `_nav<NN>.mp4` into the working directory.

---

### 3. Reproducing the campaign

#### Step 1 — Collect demonstrations

A scripted expert flies the platform and banks only episodes passing
every acceptance gate (genuine seated grasp, object retained through
transport, placement inside the box, zero table/gate/non-grasp contacts).

```bash
python collect_v2.py <seed> <n_units>        # v2 baseline + F1/F2 extensions
```

| File | Role |
|---|---|
| `collect_v2.py` | Main collector: seeded scene randomisation, expert pipeline (`nav_v2` → `pick_v2` → `place_v2`), acceptance gates, per-attempt manifest |
| `collect_airvla.py` | Shared prompts and expert primitives |
| `collect_dagger.py` | On-policy DAgger collector (Step 5b) |
| `relabel_v3.py` | Rewrites action dims 3–4 with arm-joint deltas (arm ablation) |

Collection seed family is **71000**, disjoint from every evaluation seed
family. Episodes upload to Hugging Face as they are banked; note that a
wall-clock kill loses the whole dataset (LeRobot writes its footer at
close), so size collection jobs under the wall.

#### Step 2 — Fine-tune π₀

Each configuration pins its episode list, seed and hyper-parameters in a
wrapper. All use π₀ base, batch 4, seed 1000, full fine-tune unless
stated.

| Configuration | Dataset | Init from | Steps |
|---|---|---|---|
| Base | `airvla_v2` (480-episode train split) | π₀ base | 47,500 |
| FT-C | `airvla_v21`, F1 episodes up-weighted via the episode list | π₀ base | 15,000 |
| FT-D | `airvla_v22` | FT-C | 15,000 |
| FT-D-KI | `airvla_v22`, VLM backbone frozen | FT-C | 5,000 |
| FT-DAG | `airvla_v24` (FT-C list + 160 DAgger episodes) | FT-C | 20,000, lr 5e-6→5e-7 |
| V3-arm | `airvla_v3` | π₀ base | 25,000 |
| ACT / DP | `airvla_v21` | from scratch | see wrappers |

Wrappers: `d_train_wrapper.py`, `dki_train_wrapper.py`,
`act_train_wrapper.py`, `dp_train_wrapper.py` (in this directory);
`e3ctrain_wrapper.py`, `dag_train_wrapper.py`, `e3cr_train_wrapper.py`
(cluster-side, `~/Scratch/airvla/`). Always run a 200-step smoke job
before committing to a full train.

#### Step 3 — Select a checkpoint

Checkpoints are selected by **lowest pinned-noise validation MSE** over
the saved ladder, computed by `valcurve_v2.py` — never by inspecting
closed-loop results. Pinning the noise matters: π₀ draws fresh flow
noise per call, so an unpinned comparison measures sampling variance
rather than the intervention.

#### Step 4 — Evaluate (frozen protocol)

`eval_v2.py` is the frozen harness. Every experimental condition is an
**additive, default-off flag**; with no flags the code path is
byte-identical to the frozen version, and each run records a `PROV`
line with script and dependency hashes.

```bash
# Pure policy, full frozen protocol: 60 pick + 20 nav scenes
python eval_v2.py <ckpt>/pretrained_model 60 20 --torchseed 1000 --tag base --video

# Scripted terminal servo (H1)
python eval_v2.py <ckpt>/pretrained_model 60 20 --torchseed 1000 --tag h1c --assist 0.15

# Learned terminal servo (E5) — the best system
python eval_v2.py <ckpt>/pretrained_model 60 20 --torchseed 1000 --tag e5c \
  --assist 0.15 --learned-servo e5_actor/e5_actor_final.pt
```

Protocol constants: scene seed family **97000** (paired across all
configurations), 1200 ticks per pick, 500 per nav, action horizon 50 /
execute 50, torch seed 1000. Because scenes are paired, configurations
are compared with exact McNemar tests on discordant scenes.

Experimental flags:

| Flag | Experiment |
|---|---|
| `--assist 0.15` | scripted terminal servo |
| `--learned-servo <actor.pt>` | learned terminal servo |
| `--exec 10` / `--rtc` | shortened execution horizon / real-time chunking |
| `--paraphrase` / `--synonyms` | OOD: unseen instruction wordings / unseen object nouns |
| `--oodpos --sceneseed 96000` | OOD: target outside the trained ±0.35 m band |
| `--novel-distractor` | OOD: unseen mustard bottle added as clutter |
| `--novel-target` | open-vocabulary probe: the bottle commanded as target |
| `--comp --sceneseed 95000` | held-out composite instruction (gate → pick → place) |
| `--solo --sceneseed 99000` | single-object competence probe |
| `--policy-arm` | arm ablation: policy commands the arm joints |
| `--pag` | payload-mass compensation ablation |

Run a 10+4 episode mini first — it gates the full run for harness
validity, never for conclusions (minis are too small to judge results).

#### Step 5 — Terminal controllers

**5a. Scripted servo.** Implemented in `platform_v2.py`, enabled by
`--assist 0.15`: when the jaws enter a 0.15 m capture radius of the
**instruction-named** object, control transfers to a deterministic
descend–align–close sequence. Because the trigger keys on the commanded
object, it cannot rescue a wrong-target approach — the hybrid never
conceals a grounding failure. Validate with
`replay_h1_validation.py`.

**5b. Learned servo (the best system).** A 2×128 tanh MLP mapping an
18-D body-frame observation to 4 actions:

| File | Stage |
|---|---|
| `rl_env.py` | Terminal-phase environment contract — imported, never copied, by every consumer |
| `rl_nets.py` | Actor-critic architecture |
| `rl_bc.py` | Stage 0: DAgger-clone the scripted servo (this is where success-rate learning happens: 0 → 6 → 59 welds per 60 episodes over three rounds) |
| `rl_train.py` | Stage 1: PPO refinement (71 iterations, 3,373 episodes, holding 89–100% weld rate) |
| `replay_e5_validation.py` | Contract-parity validation before results count |

Training progression is plotted in
[`Reports/e5_training_progression.png`](Reports/e5_training_progression.png).

---

### 4. Reproducibility practices used throughout

These were adopted after failures that silently corrupted earlier
results; they are worth keeping:

- **Validate the harness before the policy.** Any changed evaluation
  path is first driven with *expert* actions (`replay_*.py`,
  `comp_setcheck.py`); if the expert cannot complete the task through
  the new path, policy numbers from it are meaningless.
- **Pin sampling noise** for any comparison, or the measurement is
  sampling variance.
- **Offline metrics do not arbitrate design questions.** The arm
  ablation posted the campaign's best-looking validation curve and its
  worst closed-loop behaviour; only closed-loop evaluation decides.
- **Additive, default-off flags with provenance hashes**, so the frozen
  harness stays frozen.
- **One log file per job**, never reused — a reused log makes a gate
  read a previous run's result.
- **Gate merges on clean exit**, and never delete checkpoints with a
  globbed `rm -rf dir/*/` (it follows baseline symlinks and destroys
  their targets — this is how C-15000 was lost).

---

## Contributing & Roadmap

* ✅ Implemented: URDF/MJCF, real–sim sync, PWM/position modes, torque approx.
* 🛠 Planned: MPPI interface, torque–PWM calibration, sim–real residual compensation (Torch)

PRs and issues welcome.

---

## Authors and Contact

Zhuohang Wu @
zhuohang2024@163.com

## License

MIT (unless otherwise specified in individual files)

---

## Acknowledgements

* MuJoCo team & community
* Dynamixel SDK
* SolidWorks URDF Exporter


