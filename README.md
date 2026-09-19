# VLA Application for a Quadcopter with Manipulator for Pick-and-Place Tasks

A pretrained **π₀ vision-language-action (VLA) policy** adapted to a quadcopter
carrying a two-degree-of-freedom arm and a parallel gripper, evaluated in
MuJoCo. The policy receives three RGB camera streams, proprioception and a
written instruction, and outputs drone motion and gripper commands; the arm
follows deterministic task-phase logic.

This repository accompanies the MSc dissertation of the same title
(University College London, September 2026). It contains the simulation
environment, the demonstration collectors, the training and evaluation
pipeline, the scripted and learned terminal controllers, and the mechatronic
design files for the physical platform.

**Headline result.** The pure VLA reliably selects and approaches the commanded
object but rarely grasps it (1/60). Assigning the final 0.15 m to a specialised
terminal controller raises this to **36/60 grasps (60%) and 27/60 complete
pick-and-place (45%)**.

---

## Evidence Structure

| Evidence | Location |
|---|---|
| Experimental ledger: pre-registrations, amendments, every result with job IDs and provenance | [`Reports/finaldroneresults.md`](Reports/finaldroneresults.md) |
| Figures used in the dissertation | [`Reports/`](Reports/) (`*.png`) |
| Learned-servo training progression | [`Reports/e5_training_progression.png`](Reports/e5_training_progression.png) |
| Per-episode evaluation records | `~/Scratch/airvla/logs/*.log` (cluster). One `EVAL {json}` line per episode |
| Trajectory dumps | `~/Scratch/airvla/sim/eval_v2_traj.jsonl` (cluster). Drone and jaw pose every 3 ticks |
| Collection manifests | `v2_manifest_<seed>.jsonl`, `dag_manifest_<seed>.jsonl`. One line per attempt, accepted or rejected |
| Rendered evaluation episodes | `Reports/v2_eval_videos/`. 206 MP4s, ~4.6 GB, **git-ignored** (local only) |

The ledger is the authoritative record: it was written before each experiment
(pre-registration) and amended with dated entries, including disclosed errors
and their recovery. This README is the operational guide.

---

## Quick Start

The simulator version is **load-bearing**: all training data and every frozen
evaluation were produced under **MuJoCo 3.3.4**. A different MuJoCo changes
contact behaviour enough to invalidate comparison with the banked results.

```bash
conda create -n airvla python=3.10 -y && conda activate airvla
pip install mujoco==3.3.4 numpy scipy torch imageio imageio-ffmpeg
pip install lerobot huggingface_hub
export MUJOCO_GL=egl        # headless rendering on cluster nodes
```

Working directory for all commands below: `Sim'n'Real/Mujoco/`.

View the simulated scene and task objects:

```bash
python objects_viewer.py
```

Run a frozen evaluation of the best system:

```bash
huggingface-cli download hanapasta/airvla_ftc_15000 --local-dir ./ftc_15000
python eval_v2.py ./ftc_15000 60 20 --torchseed 1000 --tag e5c \
  --assist 0.15 --learned-servo e5_actor/e5_actor_final.pt
```

---

## Repository Layout

```
.
├─ Sim'n'Real/Mujoco/     # simulation, collectors, training wrappers, evaluation
├─ Reports/               # experimental ledger, figures, analysis
├─ cluster_jobs/          # cluster submission scripts
├─ 3D Model/              # CAD assets (SolidWorks / STL)
├─ SkyGrip_URDF/          # manipulator URDF and STL meshes
├─ v2_design/             # v2 platform design files
├─ onboard_Raspi5/        # onboard compute (Raspberry Pi 5)
├─ cam3_rerender/         # external-camera re-rendering
├─ dataset_tests/         # dataset integrity checks
└─ Medias/                # photographs and renders
```

Key files in `Sim'n'Real/Mujoco/`:

| File | Role |
|---|---|
| `SkyGrip_airvla.xml` | The MuJoCo scene used for all experiments |
| `collect_v2.py` | Main demonstration collector (expert pipeline + acceptance gates) |
| `collect_airvla.py` | Shared prompts and expert primitives |
| `collect_dagger.py` | On-policy corrective collector (FT-DAG) |
| `relabel_v3.py` | Rewrites action dims 3-4 as arm-joint deltas (arm ablation) |
| `eval_v2.py` | Frozen evaluation harness; all experiments are additive, default-off flags |
| `platform_v2.py` | Flight platform and the scripted terminal servo |
| `pd_flight.py` | Cascaded PID position / PD attitude flight controller |
| `rl_env.py`, `rl_nets.py`, `rl_bc.py`, `rl_train.py` | Learned terminal servo (contract, architecture, DAgger, PPO) |
| `objects_viewer.py` | Interactive viewer for the task objects |

---

## Simulation Environment

The MuJoCo model was built from the measured physical platform: an all-up mass
of **1097 g** (736 g airframe, 331 g two-link arm, 30 g Pololu parallel
gripper), giving a hover thrust requirement of about 10.8 N.

The scene was reconstructed photogrammetrically from the physical laboratory
(Polycam, ~1.01 × 10⁶ points) and reproduced in MuJoCo, including the 40 mm
raised working mat. A fixed table, a navigation gate and a delivery box on the
floor complete the workspace.

**Sensing** follows a three-view arrangement: a forearm-mounted camera looking
down at the grasp region, a forward-facing camera for navigation, and a fixed
external third-person view. Frames are stored at 512×512 and recorded at 10 Hz
alongside a ten-dimensional proprioceptive vector (pose, gripper aperture, both
arm joint angles).

**Flight control** is cascaded: a PID position loop whose integral term engages
only within 0.15 m of the setpoint (preventing wind-up during transit), feeding
a PD attitude loop with exact feed-forward compensation of the moment created by
the arm's offset centre of mass.

**Grasping** uses a weld constraint that activates only when the gripper closes
with validated alignment, height and aperture relative to the object, measured
from simulator state. It maintains an already-successful grasp rather than
creating one, and is released over the delivery box.

### Task Definitions

| Task | Description |
|---|---|
| **Pick and place** | Two objects on the table (blue penguin, grey calibration weight); the instruction names which to collect and place in the box |
| **Gate navigation** | Fly through a gate and hover over the named object, without grasping |
| **Compositional** | Gate traversal followed by pick-and-place in one instruction. **Never demonstrated in training**; reserved for evaluation |

Task objects: calibration weight (16 mm graspable width, 100 g) and plush
penguin (22 mm head, 45 g), against a 32 mm jaw opening, leaving 8 mm and 5 mm
of clearance per side respectively.

---

## Datasets

All datasets are in LeRobot format and hosted on Hugging Face.

| Dataset | Contents | Episodes / frames |
|---|---|---|
| [`hanapasta/airvla_v2`](https://huggingface.co/datasets/hanapasta/airvla_v2) | v2 baseline: 360 pick + 240 navigation demonstrations | 600 / 239,520 |
| [`hanapasta/airvla_v21`](https://huggingface.co/datasets/hanapasta/airvla_v21) | v2 + F1 terminal-corrective; trains **FT-C** | 910 / 391,750 |
| [`hanapasta/airvla_v22`](https://huggingface.co/datasets/hanapasta/airvla_v22) | v2 + F1 + F2 paired-command scaling; trains **FT-D / FT-D-KI** | 1,358 / 622,801 |
| [`hanapasta/airvla_v24`](https://huggingface.co/datasets/hanapasta/airvla_v24) | v21 + on-policy DAgger; trains **FT-DAG** | 1,110 / 512,577 |
| [`hanapasta/airvla_v3`](https://huggingface.co/datasets/hanapasta/airvla_v3) | v2 relabelled with arm-joint deltas; trains the arm ablation | 600 / 239,520 |
| [`airvla_dag1`](https://huggingface.co/datasets/hanapasta/airvla_dag1) to [`dag4`](https://huggingface.co/datasets/hanapasta/airvla_dag4) | Raw DAgger collection shards (merged into v24) | n/a |
| [`airvla_v2_d2`](https://huggingface.co/datasets/hanapasta/airvla_v2_d2) to [`d4`](https://huggingface.co/datasets/hanapasta/airvla_v2_d4), [`airvla_v2_e3`](https://huggingface.co/datasets/hanapasta/airvla_v2_e3) | Plan D and F1 collection shards | n/a |

```bash
huggingface-cli download hanapasta/airvla_v21 --repo-type dataset \
  --local-dir ~/hf_cache/lerobot/hanapasta/airvla_v21
# Training offline? Remove the legacy download marker or LeRobot forces a
# hub re-sync and fails under HF_HUB_OFFLINE=1:
rm -rf ~/hf_cache/lerobot/hanapasta/airvla_v21/.cache
```

---

## Trained Models

Each π₀ checkpoint is ~8.3 GB. Every public repository carries a model card
recording its training data, initialisation, step count, selection rule and
frozen-protocol result.

| Configuration | Hugging Face |
|---|---|
| **Base** (π₀, step 47500) | [`hanapasta/airvla_v2_pi0_047500`](https://huggingface.co/hanapasta/airvla_v2_pi0_047500) |
| **FT-C** (step 15000), the strongest pure policy | [`hanapasta/airvla_ftc_15000`](https://huggingface.co/hanapasta/airvla_ftc_15000) |
| **FT-D** (step 15000) | [`hanapasta/airvla_ftd_15000`](https://huggingface.co/hanapasta/airvla_ftd_15000) |
| **FT-D-KI** (step 5000, frozen backbone) | [`hanapasta/airvla_ftdki_5000`](https://huggingface.co/hanapasta/airvla_ftdki_5000) |
| **FT-DAG** (step 17500, on-policy corrective) | [`hanapasta/airvla_ftdag_17500`](https://huggingface.co/hanapasta/airvla_ftdag_17500) |
| **V3-arm** (step 25000, policy-controlled arm) | [`hanapasta/airvla_v3arm_25000`](https://huggingface.co/hanapasta/airvla_v3arm_25000) |
| **Learned terminal servo** | [`hanapasta/airvla_e5_actor`](https://huggingface.co/hanapasta/airvla_e5_actor); actor + `rl_env.py` + `rl_nets.py` |
| **ACT baseline** | [`hanapasta/act_v21`](https://huggingface.co/hanapasta/act_v21) |
| **Diffusion Policy baseline** | [`hanapasta/dp_v21`](https://huggingface.co/hanapasta/dp_v21) |

> The published FT-C checkpoint is a deterministic **retrain** of the original,
> which was destroyed by a cleanup script that followed a baseline symlink. It
> was accepted only after equivalence gates: pinned-noise validation MSE
> 7.7 × 10⁻⁵ against the original's 7.679 × 10⁻⁵, plus a matching closed-loop
> evaluation. The incident and recovery are documented in full in the ledger.
>
> Ignore `hanapasta/airvla_d_15000`, an earlier backup whose push never
> completed; it holds only `.gitattributes` and is superseded by
> `airvla_ftd_15000`.

---

## Evaluation Protocol

`eval_v2.py` is frozen. Every experimental condition is an **additive,
default-off flag**; with no flags the code path is byte-identical to the frozen
version, and each run records a `PROV` line with script and dependency hashes.

| Setting | Value |
|---|---|
| Pick-and-place | 60 episodes, maximum 1200 control steps |
| Navigation | 20 episodes, maximum 500 control steps |
| Evaluation scenes | Seed family 97000, disjoint from collection and development probes |
| Paired comparison | Every configuration sees the same scene at a given episode index |
| Default execution | Predict H = 50 actions, execute E = 50 before replanning |
| Torch seed | 1000 |

Because scenes are paired, configurations are compared with **exact McNemar
tests** on discordant scenes, Wilcoxon signed-rank on per-scene distances, and
Wilson intervals on proportions.

### Experiment Flags

| Flag | Experiment |
|---|---|
| `--assist 0.15` | Scripted terminal servo |
| `--learned-servo <actor.pt>` | Learned terminal servo |
| `--exec 10` / `--rtc` | Shortened execution horizon / Real-Time Chunking |
| `--paraphrase` / `--synonyms` | OOD: unseen instruction wordings / unseen object names |
| `--oodpos --sceneseed 96000` | OOD: target outside the trained ±0.35 m band |
| `--novel-distractor` | OOD: unseen mustard bottle added as clutter |
| `--novel-target` | Open-vocabulary probe: the bottle commanded as target |
| `--comp --sceneseed 95000` | Held-out composite instruction |
| `--solo --sceneseed 99000` | Single-object competence probe |
| `--policy-arm` | Ablation: policy commands the arm joints |
| `--pag` | Payload-mass compensation ablation |

Run a 10+4 episode mini first; it gates the full run for harness validity,
never for conclusions.

---

## Reproducing the Experiments

**1. Collect demonstrations.** A scripted expert flies the platform and banks
only episodes passing every acceptance gate (genuine seated grasp, object
retained through transport, placement inside the box, zero table/gate/non-grasp
contacts). Collection seed family is 71000, disjoint from every evaluation seed.

```bash
python collect_v2.py <seed> <n_units>
```

Size collection jobs under the wall clock: a wall-killed job loses the entire
dataset, because LeRobot writes its footer only at close.

**2. Fine-tune π₀.** All configurations use π₀ base, batch 4, seed 1000, full
fine-tune unless stated.

| Configuration | Dataset | Initialised from | Steps |
|---|---|---|---|
| Base | `airvla_v2` (480-episode training split) | π₀ base | 47,500 |
| FT-C | `airvla_v21`, F1 up-weighted via the episode list | π₀ base | 15,000 |
| FT-D | `airvla_v22` | FT-C | 15,000 |
| FT-D-KI | `airvla_v22`, VLM backbone frozen | FT-C | 5,000 |
| FT-DAG | `airvla_v24` | FT-C | 20,000, lr 5e-6→5e-7 |
| V3-arm | `airvla_v3` | π₀ base | 25,000 |
| ACT / DP | `airvla_v21` | from scratch | 50,000 |

Wrappers: `d_train_wrapper.py`, `dki_train_wrapper.py`, `act_train_wrapper.py`,
`dp_train_wrapper.py`. Always run a 200-step smoke job first.

**3. Select a checkpoint.** By **lowest pinned-noise validation MSE** over the
saved ladder (`valcurve_v2.py`), never by inspecting closed-loop results.
Pinning the noise matters: π₀ draws fresh flow noise per call, so an unpinned
comparison measures sampling variance rather than the intervention.

**4. Evaluate.** See the protocol above.

**5. Terminal controllers.** The **scripted servo** (`platform_v2.py`,
`--assist 0.15`) hands off when the jaws enter a 0.15 m capture radius of the
*instruction-named* object, so it cannot rescue a wrong-target approach. The
**learned servo** is a 2×128 tanh MLP mapping an 18-D body-frame observation to
4 actions, trained by DAgger cloning of the scripted closer (`rl_bc.py`)
followed by PPO refinement (`rl_train.py`).

---

## Included Results

Frozen protocol, n = 60 pick-and-place + 20 navigation, paired scenes.

| Configuration | Correct target | Grasped | Placed | Median (mm) | Nav. |
|---|---|---|---|---|---|
| Base (π₀ 47500) | 42/60 | 1/60 | 1/60 | 170.8 | 9/20 |
| FT-C | 47/60 | 1/60 | 1/60 | 145.8 | 12/20 |
| FT-D | 40/60 | 1/60 | 1/60 | 204.9 | 10/20 |
| FT-D-KI | 43/60 | 0/60 | 0/60 | 117.8 | 12/20 |
| FT-DAG | 37/60 | 1/60 | 1/60 | 196.0 | 12/20 |
| ACT | 28/60 | 1/60 | 1/60 | 313.5 | 10/20 |
| Diffusion Policy | 35/60 | 0/60 | 0/60 | 249.8 | 0/20 |
| Base + Scripted | 41/60 | 21/60 | 19/60 | 18.2 | 10/20 |
| FT-C + Scripted | 46/60 | 24/60 | 22/60 | 19.2 | 11/20 |
| **FT-C + Learned** | 45/60 | **36/60** | **27/60** | **12.8** | 11/20 |
| FT-D-KI + Learned | 46/60 | 33/60 | 26/60 | 14.9 | 12/20 |

Adding the learned terminal controller to FT-C is significant on the paired
scenes (exact McNemar *p* = 5.8 × 10⁻¹¹ for grasping, *p* = 3.0 × 10⁻⁸ for
placement), while target selection and navigation are unchanged, so the gain is
specific to terminal conversion.

**Out-of-distribution** (FT-C + Learned): performance is unchanged under
paraphrased instructions (36/60 grasps), unseen object names (37/60) and a novel
visual distractor (33/60), but falls under spatial extrapolation (18/60, target
outside the trained band). Target-true precision on reached targets stays at
11.2 mm, so the failure is in *acquiring* out-of-range targets, not in terminal
control.

**Composite task** (never demonstrated): both configurations cross the gate in
60/60 episodes. Pure FT-C achieves no grasps; the hybrid achieves 24/60 grasps
and **3/60 complete composite successes**, with post-grasp carry the limiting
stage.

Full results, statistics and provenance:
[`Reports/finaldroneresults.md`](Reports/finaldroneresults.md).

### Compute Cost

Single **NVIDIA A100-PCIE-40GB** per job (UCL Myriad).

| Stage | Cost |
|---|---|
| v2 collection (600 episodes) | 17.8 h |
| F1 collection (310 episodes) | 7.5 h |
| Base training (60,000 steps) | ~22.5 h |
| Each fine-tune (20,000 steps) | ~8 h (5 h 13 m with the backbone frozen) |
| Learned-servo PPO (71 iterations) | ~8 h |
| One frozen evaluation (60 + 20) | ~2 h |

---

## Physical Platform

The physical testbed uses a Volador II VX6 frame powered by a 4S-6S LiPo
battery, with a KM60A BLHeli-32 ESC and four 2207, 1900 kV brushless motors.
Flight control, sensor fusion and low-level stabilisation run on a Paparazzi
Tawaki V2 board; high-level processing runs on a Raspberry Pi 5. A two-DoF arm
is mounted beneath the airframe with a motorised parallel gripper at its
end-effector.

| Component | Mass |
|---|---:|
| Frame, ESCs, motors, Raspberry Pi 5 | 386 g |
| Battery | 350 g |
| *Subtotal (base airframe)* | *736 g* |
| Manipulator (two links) | 331 g |
| Gripper (Pololu Micro Gripper Kit) | 30 g |
| **Total all-up mass** | **1097 g** |

**Platform development.** Four aerial systems were assembled and brought up at
UCL East, including soldering, flight-board checks and motor/thrust
verification. The original four-claw gripper (~100 g) obstructed the
wrist-camera view of the grasp region and was replaced with a 30 g Pololu Micro
Gripper Kit with position-feedback servo; a custom adapter was designed to
connect it to the existing arm, and the landing legs were shortened to reduce
collision risk during manipulation. Together these reduced end-effector mass by
about 70 g and cleared the view of the grasp region. Design files are in
[`3D Model/`](3D%20Model/) and [`SkyGrip_URDF/`](SkyGrip_URDF/).

**The learned policy was never deployed on hardware.** Every result in this
repository is from simulation. See Research Boundary.

---

## Reproducibility Practices

Adopted after failures that silently corrupted earlier results:

- **Validate the harness before the policy.** Any changed evaluation path is
  first driven with *expert* actions (`replay_*.py`, `comp_setcheck.py`); if the
  expert cannot complete the task through it, policy numbers from it are
  meaningless.
- **Pin sampling noise** for any comparison, or the measurement is sampling
  variance.
- **Offline metrics do not arbitrate design questions.** The arm ablation posted
  the best-looking validation curve and the worst closed-loop behaviour.
- **Additive, default-off flags with provenance hashes**, so the frozen harness
  stays frozen.
- **One log file per job**, never reused, because a reused log makes a gate read a
  previous run's result.
- **Never delete checkpoints with a globbed `rm -rf dir/*/`**, because it follows
  baseline symlinks and destroys their targets. Enumerate with
  `find -maxdepth 1 -type d` instead, and push every selected checkpoint to the
  hub at selection time.

---

## Documentation

| Document | Contents |
|---|---|
| [`Reports/finaldroneresults.md`](Reports/finaldroneresults.md) | Experimental ledger: pre-registrations, results, statistics, incidents |
| [`Sim'n'Real/Mujoco/PROVENANCE.md`](Sim'n'Real/Mujoco/PROVENANCE.md) | File provenance and hashes |
| [`Sim'n'Real/Mujoco/INTERPRETABILITY_README.md`](Sim'n'Real/Mujoco/INTERPRETABILITY_README.md) | Interpretability probes |
| [`Sim'n'Real/Mujoco/SAGEMAKER_TRAINING.md`](Sim'n'Real/Mujoco/SAGEMAKER_TRAINING.md) | Alternative training path |

---

## Research Boundary

These results establish behaviour **in simulation only**, under the frozen
protocol described above. Three constraints bound every claim:

- **No physical deployment.** The hardware exists and was brought up, but the
  learned system has never flown. Sim-to-real performance is unknown.
- **Privileged terminal-controller inputs.** Both terminal controllers read
  simulator state rather than onboard sensing. The scripted controller uses the
  target position, the learned controller an aim-to-jaw vector derived from
  ground truth. Their gains do not demonstrate a perception-driven system.
- **Simplified grasp retention.** A weld constraint holds the object after a
  validated grasp, simplifying contact and slip behaviour and potentially
  overestimating post-grasp reliability.

Each configuration was trained once, so differences between configurations
cannot be attributed to the intervention alone; the analysis is restricted to
effects large enough to be robust to that, and comparisons within seed-variance
range are reported as non-significant.

---

## Licence Status

MIT, unless otherwise specified in individual files. The `π₀` checkpoint,
LeRobot and MuJoCo carry their own licences.

---

## Author

**Hana Emma Hadidi**, MSc Artificial Intelligence for Sustainable Development,
Department of Computer Science, University College London. Supervised by
**Dr Valerio Modugno**.

This work comprises the MuJoCo simulation environment, the scripted expert and
demonstration collectors, the flight controller, the grasp and delivery
protocol, all dataset campaigns, the π₀ fine-tuning and evaluation pipeline, the
scripted and learned terminal controllers, and the analysis reported in
[`Reports/finaldroneresults.md`](Reports/finaldroneresults.md).

## Acknowledgements

With thanks to the Department of Computer Science at UCL East and Holistic AI
for the facilities and computational resources that supported this research, and
to those at UCL East who helped with the development and preparation of the
physical drone platform.

The aerial manipulator hardware (the drone airframe and two-link arm) was
inherited from a previous student project and was extended here with a
replacement gripper, a custom gripper-to-arm adapter and shortened landing legs.

Built on MuJoCo, LeRobot and the openly released π₀ checkpoint.
