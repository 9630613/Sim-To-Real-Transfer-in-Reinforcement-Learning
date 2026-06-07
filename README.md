# Sim-to-Real Transfer in Robot Learning
### A Reinforcement Learning Approach using MuJoCo Hopper

This project explores reinforcement learning (RL) for robotic control, with a focus on the **sim-to-real transfer** challenge. Policies are trained in the MuJoCo Hopper environment using the **Soft Actor-Critic (SAC)** algorithm, and generalization is evaluated using **Uniform Domain Randomization (UDR)** and **Adaptive Domain Randomization (ADR)**.


## Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Domains](#domains)
- [Algorithms](#algorithms)
- [Domain Randomization](#domain-randomization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [References](#references)


## Overview

Transferring RL policies from simulation to the real world is one of the core challenges in robotics. Simulators allow fast, safe, and cheap data collection, but policies trained in simulation often fail when deployed on real hardware due to the **reality gap** — differences in physics, dynamics, and sensor noise.

This project investigates domain randomization as a strategy to bridge this gap, using the **MuJoCo Hopper** as the test platform.

## Environment

The **Hopper** is a 2D one-legged robot simulated in MuJoCo, consisting of four rigid body parts connected by three hinge joints:

```
Torso
  └── Thigh  (hip joint)
        └── Leg  (knee joint)
              └── Foot  (ankle joint)
```

The entire body balances on a single foot. The agent must learn to apply coordinated torques across the three joints to produce stable, forward-moving hops — a task that requires balancing propulsion, stability, and energy efficiency simultaneously.

### Body Parts & Masses

| Body Part | Default Mass | Role |
|-----------|-------------|------|
| Torso | 3.534 kg | Main body, carries all other segments |
| Thigh | 3.927 kg | Upper leg, controls vertical lift |
| Leg | 2.714 kg | Lower leg, drives propulsion |
| Foot | 5.089 kg | Ground contact, balance and landing |

### State Space (11-dimensional)

The observation vector gives the agent a full picture of its current posture and motion:

| # | Observation | Description |
|---|-------------|-------------|
| 1 | `rootz` | Height of the torso above ground |
| 2 | `rooty` | Angle of the torso (forward tilt) |
| 3 | `thigh_joint` | Thigh joint angle |
| 4 | `leg_joint` | Knee joint angle |
| 5 | `foot_joint` | Ankle joint angle |
| 6 | `rootx_vel` | Horizontal (forward) velocity |
| 7 | `rootz_vel` | Vertical velocity |
| 8 | `rooty_vel` | Torso angular velocity |
| 9 | `thigh_vel` | Thigh angular velocity |
| 10 | `leg_vel` | Knee angular velocity |
| 11 | `foot_vel` | Ankle angular velocity |

### Action Space

Continuous: `Box(-1, 1, (3,), float32)` — normalized torques applied to three joints, scaled internally to environment limits:

| Joint | Role |
|-------|------|
| **Thigh joint** | Controls vertical lift and stability |
| **Leg joint** | Drives forward propulsion, absorbs landing impact |
| **Foot joint** | Maintains balance and ground contact |

### Reward Function

The reward at each timestep balances three competing objectives:

```
reward = healthy_reward + forward_reward − control_cost
```

| Component | Purpose |
|-----------|---------|
| `healthy_reward` | Fixed bonus (+1) at every timestep the hopper stays upright — encourages survival |
| `forward_reward` | Proportional to forward velocity — encourages fast movement |
| `control_cost` | Penalizes large torques (`0.001 × ||action||²`) — encourages smooth, efficient control |

### Episode Termination

An episode ends under any of the following conditions:

- **Fall detected**: torso height drops below threshold, or torso angle exceeds safe range
- **Time limit**: 1000 timesteps reached
- When `terminate_when_unhealthy=True` (used in this project), the episode ends immediately upon instability rather than continuing until the time limit


## Domains

This project defines four simulation domains to study how training environment choice affects transfer performance. All domains share the same Hopper morphology; only the physical parameters differ.

### Source Domain
The **source domain** simulates a robot with a manufacturing defect: the torso mass is set to **2.534 kg** (1 kg lighter than the real robot). All other body parts use default masses. This is the "imperfect simulator" scenario — the agent learns a policy that is well-tuned for a robot that doesn't quite match reality.

- Torso mass: **2.534 kg** (default − 1 kg)
- Thigh / Leg / Foot: default masses
- Represents: a miscalibrated or approximate simulator

### Target Domain
The **target domain** represents the real-world robot, with the torso at its true mass of **3.534 kg**. Policies are evaluated here to measure how well they transfer. Training directly on the target is impractical in real robotics (safety risks, cost, slow resets), so it is used only as a test environment.

- Torso mass: **3.534 kg** (default)
- Thigh / Leg / Foot: default masses
- Represents: the real robot / deployment environment

### UDR Domain (Uniform Domain Randomization)
The **UDR domain** builds on the source domain by randomizing the masses of the thigh, leg, and foot at the start of every episode. This creates a distribution of environments, forcing the agent to learn a robust policy that works across a range of physical configurations rather than overfitting to a single one.

- Torso mass: fixed at **2.534 kg** (source offset preserved)
- Thigh / Leg / Foot: sampled each episode from `Uniform(base × 0.8, base × 1.2)` — i.e., ±20% of default
- Randomization applied at episode reset via `env.reset()`
- Mass values stay fixed for the duration of each episode

### ADR Domain (Adaptive Domain Randomization)
The **ADR domain** extends UDR by automatically adjusting how wide the randomization ranges are based on how well the agent is performing. Rather than using a fixed ±20% range throughout training, ADR starts easy and gradually increases difficulty as the agent improves.

- Torso mass: fixed at **2.534 kg** (source offset preserved)
- Thigh / Leg / Foot: adaptively randomized — ranges start at ±5% and expand up to ±20%+
- Success threshold for expansion: mean reward > 200 over last 100 episodes
- Contraction triggered when mean reward < 100
- Minimum range floor: ±10% (prevents ranges collapsing to zero)

### Domain Comparison

| Property | Source | Target | UDR | ADR |
|----------|--------|--------|-----|-----|
| Torso mass | 2.534 kg | 3.534 kg | 2.534 kg | 2.534 kg |
| Other masses | Fixed | Fixed | Randomized (±20%) | Adaptive (±5% → ±20%+) |
| Randomization | None | None | Uniform, fixed range | Curriculum-based |
| Purpose | Train baseline | Evaluate transfer | Improve robustness | Targeted robustness |


## Algorithms

### Why SAC over PPO?

Two algorithms were evaluated — **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)**:

| Feature | PPO | SAC |
|---------|-----|-----|
| Policy type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Exploration | Noise/clipping | Entropy maximization |
| Data reuse | No | Yes (replay buffer) |

**SAC was chosen** because it is more sample efficient, reuses past experiences via a replay buffer, and inherently encourages exploration through entropy maximization — all important properties for the complex Hopper environment.

### SAC Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-4 |
| Buffer size | 100,000 |
| Batch size | 128 |
| Learning starts | 500 steps |
| Train frequency | Every 4 steps |
| Gradient steps | 1 |
| Tau (soft update) | 0.01 |

## Domain Randomization

### Uniform Domain Randomization (UDR)

- At the start of each episode, the masses of the **thigh, leg, and foot** are sampled from a uniform distribution within ±20% of their base values.
- The torso mass remains fixed with the +1 kg source offset.
- This forces the agent to generalize over a wide distribution of physical dynamics.

### Adaptive Domain Randomization (ADR)

ADR automatically adjusts randomization ranges based on agent performance:

- **Starts** with narrow ranges (±5% of base mass)
- **Expands** ranges when success rate > 0.6 (agent performs well)
- **Contracts** ranges when the agent struggles
- **Minimum range** of ±10% is maintained to prevent over-narrowing
- Performance is tracked over a 100-episode sliding window


## Results

Models were trained in all four domains and evaluated on both source and target domains.

| Trained On | Tested On Source | Tested On Target |
|------------|-----------------|-----------------|
| Source | 1180.61 ± 48.50 | 1070.05 ± 5.64 |
| Target | — | 1635.82 ± 9.20 |
| **UDR** | **1731.05 ± 5.02** | **1764.76 ± 5.90** |
| **ADR** | **1687.36 ± 5.63** | **1752.47 ± 34.90** |

**Key findings:**
- The source model shows a clear performance drop on the target domain, confirming the reality gap.
- Both UDR and ADR significantly outperform direct source training on the target domain.
- UDR and ADR achieve higher rewards even on the source domain, suggesting that randomization also improves overall robustness.
- ADR shows higher variance on the target domain, suggesting it is more sensitive to its hyperparameters.
<p align="center">
  <img src="https://github.com/9630613/Sim-To-Real-Transfer-in-Reinforcement-Learning/blob/main/Simulation_results/Source_on_target.gif?raw=true" alt="Source on Target simulation"/>
  <br/>
  <em>Source on Target simulation</em>
</p>


<p align="center">
  <img src="https://github.com/9630613/Sim-To-Real-Transfer-in-Reinforcement-Learning/blob/main/Simulation_results/Target_on_target.gif?raw=true" alt="Source on Target simulation"/>
  <br/>
  <em>Source on Target simulation</em>
</p>


<p align="center">
  <img src="https://github.com/9630613/Sim-To-Real-Transfer-in-Reinforcement-Learning/blob/main/Simulation_results/UDR_on_target.gif?raw=true" alt="Source on Target simulation"/>
  <br/>
  <em>Source on Target simulation</em>
</p>

## Project Structure

```
.
├── custom_hopper/          # Custom Hopper environment with UDR and ADR support
├── train.py                # Training script
├── test.py                 # Evaluation script
├── policy.py               # Policy definitions (SAC)
├── models/                 # Trained model checkpoints (4 models)
│   ├── source_model
│   ├── target_model
│   ├── udr_model
│   └── adr_model
├── colab_policy.py         # Colab-compatible training notebook
└── report.pdf              # Full project report
```


## Installation & Usage

> **Note:** This project was developed and tested using Google Colab due to MuJoCo installation complexity. Local setup instructions are provided below for reference.

### Dependencies

```bash
pip install stable-baselines3 gymnasium mujoco
```

### Training

```bash
python train.py --domain source    # Train on source domain
python train.py --domain target    # Train on target domain
python train.py --domain udr       # Train with Uniform Domain Randomization
python train.py --domain adr       # Train with Adaptive Domain Randomization
```

### Evaluation

```bash
python test.py --model models/udr_model --test-domain target
```


## References

1. R. Sutton and A. Barto, *Reinforcement Learning: An Introduction*. MIT Press, 1998.
2. J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," *arXiv:1703.06907*, 2017.
3. X. B. Peng et al., "Sim-to-real transfer of robotic control with dynamics randomization," *IEEE ICRA*, 2018.


*Project by Zahra Sadeghi Jalalabadi — Politecnico di Torino*
