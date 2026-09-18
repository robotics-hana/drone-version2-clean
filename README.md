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

## AirVLA Pipeline (v2 campaign)

The MSc-dissertation experiments — a π₀ vision-language-action policy
flying the SkyGrip drone + 2-DoF arm in MuJoCo — live in
`Sim'n'Real/Mujoco/`. The four stages below reproduce the system;
results and full provenance are tracked in
`Reports/finaldroneresults.md`.

### 1. Expert demonstrations (training data)

The scripted expert pilots the platform and banks accepted episodes
as a LeRobot dataset:

| File | Role |
|---|---|
| `collect_v2.py` | Main v2 collector: seeded scene randomisation, the expert pipeline (`nav_v2` → `pick_v2` → `place_v2`), acceptance gates (genuine seated grasp, zero table/gate contacts, placement inside the box), per-attempt manifest. Modes for standard picks, corrective picks and nav episodes. |
| `collect_airvla.py` | Shared prompts and expert primitives used by every collector. |
| `collect_dagger.py` | FT-DAG on-policy collector: the fine-tuned VLA rolls in (unrecorded); the scripted expert takes over at a sustained-near seam and completes the episode — only expert actions are banked. |

Run on a GPU node in the pinned environment, e.g.
`python collect_v2.py <seed> <n_units>` (see each file's docstring
for arguments). Episodes upload to Hugging Face datasets
(`hanapasta/airvla_v2*`).

### 2. VLA training (π₀ fine-tune)

Training uses [LeRobot](https://github.com/huggingface/lerobot)'s
π₀ implementation on the UCL Myriad cluster; each configuration has
a wrapper that pins the episode list, seed and hyper-parameters:

| File | Configuration |
|---|---|
| cluster `v2train.job` (+ wrapper) | v2 Base: full fine-tune of π₀ on the 480-episode training split, batch 4, seed 1000. |
| cluster `e3ctrain_wrapper.py` | FT-C: v2+F1 with rebalanced sampling via the episode list. |
| `d_train_wrapper.py` / `dki_train_wrapper.py` | FT-D (from FT-C) and FT-D-KI (frozen VLM backbone). |
| cluster `dag_train_wrapper.py` | FT-DAG: FT-C list + on-policy corrective episodes, initialised from FT-C. |
| `act_train_wrapper.py` / `dp_train_wrapper.py` | From-scratch ACT and Diffusion Policy baselines. |

Checkpoints are selected by the lowest pinned-noise validation MSE
over the saved ladder (`valcurve` jobs), never by peeking at
closed-loop results. Evaluation runs through the frozen harness
`eval_v2.py` (additive, default-off flags only; every run records a
PROV line).

### 3. Scripted terminal servo ("Scripted hybrid", H1)

The scripted closer lives in `platform_v2.py` and is enabled at
evaluation time with `--assist 0.15`: when the jaws come within the
0.15 m capture radius of the *commanded* object, control hands off
to a deterministic terminal descend-align-close sequence. Because
the trigger uses the instruction-named object, it cannot convert a
wrong-target approach. `replay_h1_validation.py` replays expert
episodes through the assist path to validate the handoff.

### 4. Learned terminal servo ("Learned hybrid", E5)

A small MLP (2×128 tanh, 18-D body-frame observation → 4 actions)
replaces the scripted closer:

| File | Role |
|---|---|
| `rl_env.py` | The terminal-phase environment contract (observation/action/episode definitions — imported, not copied, by every consumer). |
| `rl_nets.py` | Actor-critic architecture. |
| `rl_bc.py` | Stage 0: DAgger-clones the scripted servo into the actor (the 62%→90% conversion mechanism). |
| `rl_train.py` | Stage 1: PPO polish against `rl_env.TerminalEnv`. |
| `e5_actor/e5_actor_final.pt` | The final trained actor (also on HF: `hanapasta/airvla_e5_actor`). |
| `replay_e5_validation.py` | Contract-parity validation before results count. |

Deploy with
`python eval_v2.py <ckpt> 60 20 --torchseed 1000 --assist 0.15 --learned-servo e5_actor/e5_actor_final.pt`.
The best system (FT-C + learned servo, "E5C") reaches 60% grasp /
45% full pick-and-place on the frozen n=60 protocol.

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


