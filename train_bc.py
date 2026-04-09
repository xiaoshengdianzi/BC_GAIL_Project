import random
import argparse
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


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions) + 1e-8)
        bc_loss = torch.mean(-log_probs)
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()
        return bc_loss.item()

    def take_action(self, state):
        state = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


def test_agent(agent, env, n_episode, seed=2000):
    returns = []
    for i in range(n_episode):
        state = rl_utils.reset_env(env, seed=seed + i)
        done = False
        ep_return = 0.0
        while not done:
            action = agent.take_action(state)
            state, reward, done, _ = rl_utils.step_env(env, action)
            ep_return += reward
        returns.append(ep_return)
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning on CartPole expert dataset.")
    parser.add_argument("--data", type=str, default="expert_data_cartpole.npz", help="Path to expert .npz dataset.")
    parser.add_argument("--n_iterations", type=int, default=1000, help="BC training iterations.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for BC updates.")
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    data = np.load(args.data)
    expert_states = data["states"]
    expert_actions = data["actions"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bc_agent = BehaviorClone(
        state_dim=state_dim,
        hidden_dim=128,
        action_dim=action_dim,
        lr=1e-3,
        device=device,
    )

    n_iterations = args.n_iterations
    batch_size = args.batch_size
    test_returns = []

    with tqdm(total=n_iterations, desc="BC progress") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0, high=expert_states.shape[0], size=batch_size)
            loss = bc_agent.learn(expert_states[sample_indices], expert_actions[sample_indices])

            current_return = test_agent(bc_agent, env, n_episode=5)
            test_returns.append(current_return)

            if (i + 1) % 10 == 0:
                pbar.set_postfix(
                    {
                        "return": f"{np.mean(test_returns[-10:]):.3f}",
                        "loss": f"{loss:.4f}",
                    }
                )
            pbar.update(1)

    score = test_agent(bc_agent, env, n_episode=20)
    print(f"BC mean return (20 eps): {score:.2f}")
    print(f"Used dataset file: {args.data}")

    torch.save(bc_agent.policy.state_dict(), "bc_policy_cartpole.pth")
    print("Saved BC policy to bc_policy_cartpole.pth")

    iteration_list = list(range(len(test_returns)))
    plt.figure(figsize=(8, 5))
    plt.plot(iteration_list, test_returns)
    plt.xlabel("Iterations")
    plt.ylabel("Returns")
    plt.title("BC on CartPole-v0")
    plt.tight_layout()
    plt.savefig("bc_returns_curve.png", dpi=150)
    print("Saved return curve to bc_returns_curve.png")


if __name__ == "__main__":
    main()
