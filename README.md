# Imitation Learning Implementations

This repository contains implementations of various imitation learning algorithms in PyTorch, including Behavioral Cloning (BC) and Generative Adversarial Imitation Learning (GAIL), along with expert policy training using Proximal Policy Optimization (PPO).

## Environment

- Python 3.8+
- PyTorch
- gymnasium (preferred) or gym (fallback)
- numpy, tqdm, matplotlib

## Install

```bash
pip install torch numpy tqdm matplotlib gymnasium
```

If you use gym (older installs), replace gymnasium with gym.

## Run

### 1. Train Expert Policy with PPO
```bash
python train_expert_ppo.py
```

### 2. Train Behavioral Cloning (BC)
```bash
python train_bc.py
```

### 3. Train Generative Adversarial Imitation Learning (GAIL)
```bash
python train_gail.py
```

## Results

The following files are generated during training:

- `expert_actor_ppo_cartpole.pth` - Trained expert PPO policy
- `bc_policy_cartpole.pth` - Trained BC policy
- `gail_policy_actor.pth` - Trained GAIL policy
- `bc_returns_curve.png` - BC training returns curve
- `gail_returns_curve.png` - GAIL training returns curve
- `expert_data_*.npz` - Expert demonstration data

## Project Structure

- `train_expert_ppo.py` - Train expert policy using PPO
- `train_bc.py` - Train Behavioral Cloning policy
- `train_gail.py` - Train GAIL policy
- `rl_utils.py` - Utility functions for reinforcement learning
- `*.pth` files - Trained policy weights
- `*.npz` files - Expert demonstration data
- `*.png` files - Training result plots

## Algorithms

### 1. Proximal Policy Optimization (PPO)
- Used to train expert policies that generate demonstration data
- Implements the PPO algorithm with clipped objective

### 2. Behavioral Cloning (BC)
- Directly learns a policy from expert demonstrations
- Simple supervised learning approach

### 3. Generative Adversarial Imitation Learning (GAIL)
- Uses a generative adversarial network to learn from expert demonstrations
- Consists of a policy network and a discriminator network

