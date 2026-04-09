import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import rl_utils
from train_expert_ppo import PPO


class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.action_dim = action_dim
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.long).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.long).to(self.device)

        expert_actions = F.one_hot(expert_actions, num_classes=self.action_dim).float()
        agent_actions = F.one_hot(agent_actions, num_classes=self.action_dim).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        discriminator_loss = (
            nn.BCELoss()(agent_prob, torch.ones_like(agent_prob))
            + nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        )

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = (-torch.log(agent_prob + 1e-8)).detach().cpu().numpy().reshape(-1)

        transition_dict = {
            "states": agent_s,
            "actions": agent_a,
            "rewards": rewards,
            "next_states": next_s,
            "dones": dones,
        }
        self.agent.update(transition_dict)
        return float(discriminator_loss.item())


def main():
    parser = argparse.ArgumentParser(description="GAIL training on CartPole with PPO policy optimization.")
    parser.add_argument("--expert_data", type=str, default="expert_data_30.npz", help="Expert data .npz file.")
    parser.add_argument("--n_episode", type=int, default=500, help="Training episodes for GAIL.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer width.")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="Discriminator learning rate.")
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "CartPole-v0"
    env = gym.make(env_name)

    data = np.load(args.expert_data)
    expert_s = data["states"]
    expert_a = data["actions"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(
        state_dim,
        args.hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )
    gail = GAIL(agent, state_dim, action_dim, args.hidden_dim, args.lr_d, device)

    return_list = []
    discriminator_loss_list = []

    with tqdm(total=args.n_episode, desc="GAIL progress") as pbar:
        for i in range(args.n_episode):
            episode_return = 0.0
            state = rl_utils.reset_env(env, seed=seed + i)
            done = False

            state_list = []
            action_list = []
            next_state_list = []
            done_list = []

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = rl_utils.step_env(env, action)

                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)

                state = next_state
                episode_return += reward

            return_list.append(episode_return)
            d_loss = gail.learn(expert_s, expert_a, state_list, action_list, next_state_list, done_list)
            discriminator_loss_list.append(d_loss)

            if (i + 1) % 10 == 0:
                pbar.set_postfix(
                    {
                        "return": f"{np.mean(return_list[-10:]):.3f}",
                        "d_loss": f"{np.mean(discriminator_loss_list[-10:]):.4f}",
                    }
                )
            pbar.update(1)

    print(f"GAIL final mean return (last 20 eps): {np.mean(return_list[-20:]):.2f}")

    iteration_list = list(range(len(return_list)))
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"GAIL on {env_name} ({args.expert_data})")
    plt.tight_layout()
    plt.savefig("gail_returns_curve.png", dpi=150)
    print("Saved return curve to gail_returns_curve.png")

    torch.save(agent.actor.state_dict(), "gail_policy_actor.pth")
    print("Saved GAIL policy actor to gail_policy_actor.pth")


if __name__ == "__main__":
    main()
