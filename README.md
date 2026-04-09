# Imitation Learning Implementations

<div align="center">
  <img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=professional%20imitation%20learning%20ai%20robot%20training%20neural%20network%20visualization&image_size=square_hd" alt="Imitation Learning" width="400">
</div>

<div align="center">
  <img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=imitation%20learning%20behavioral%20cloning%20framework%20diagram%20expert%20demonstrations%20policy%20network%20supervised%20learning&image_size=landscape_16_9" alt="Imitation Learning Framework" width="600">
</div>

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/xiaoshengdianzi/BC_GAIL_Project.git
```

### Install Dependencies

You can install dependencies directly:

```bash
pip install torch numpy tqdm matplotlib gymnasium
```

Or from the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── train_expert_ppo.py   # PPO expert policy training
├── train_bc.py           # Behavioral Cloning implementation
├── train_gail.py         # GAIL implementation
├── rl_utils.py           # Reinforcement learning utilities
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
└── README.md             # This documentation
```

## Results

### Training Curves

<div align="center">
  <h4>BC Training Curve</h4>
  <img src="bc_returns_curve.png" alt="BC Training Curve" width="500">

  <h4>GAIL Training Curve</h4>
  <img src="gail_returns_curve.png" alt="GAIL Training Curve" width="500">
</div>

## Overview

This repository provides implementations of imitation learning algorithms in PyTorch, including Behavioral Cloning (BC) and Generative Adversarial Imitation Learning (GAIL), with expert policy training using Proximal Policy Optimization (PPO).

## Algorithms

### 1. Proximal Policy Optimization (PPO)
Used to train expert policies that generate high-quality demonstration data.

### 2. Behavioral Cloning (BC)
A simple approach that directly learns a policy from expert demonstrations using supervised learning.

### 3. Generative Adversarial Imitation Learning (GAIL)
Uses a GAN framework to learn from expert demonstrations without explicitly specifying a reward function.

## Usage

### Train Expert Policy with PPO
```bash
python train_expert_ppo.py --sample_episodes 1 --n_samples 30
```

### Train Behavioral Cloning (BC)
```bash
python train_bc.py
```

### Train Generative Adversarial Imitation Learning (GAIL)
```bash
python train_gail.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.