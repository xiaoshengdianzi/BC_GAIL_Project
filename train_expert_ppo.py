import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import rl_utils


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    ):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def sample_expert_data(env, ppo_agent, n_episode, seed=0):
    states = []
    actions = []
    for episode in range(n_episode):
        state = rl_utils.reset_env(env, seed=seed + episode)
        done = False
        while not done:
            action = ppo_agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, _, done, _ = rl_utils.step_env(env, action)
            state = next_state
    return np.array(states), np.array(actions)


def evaluate_policy(env, policy, n_episodes=10, seed=1000):
    returns = []
    for i in range(n_episodes):
        state = rl_utils.reset_env(env, seed=seed + i)
        done = False
        ep_return = 0.0
        while not done:
            action = policy.take_action(state)
            state, reward, done, _ = rl_utils.step_env(env, action)
            ep_return += reward
        returns.append(ep_return)
    return float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser(description="Train PPO expert and export BC dataset.")
    parser.add_argument("--sample_episodes", type=int, default=1, help="Episodes used to collect expert trajectories.")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of (s, a) pairs sampled for BC.")
    parser.add_argument("--output_npz", type=str, default="expert_data_cartpole.npz", help="Output expert dataset path.")
    args = parser.parse_args()

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 250
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    env_name = "CartPole-v0"

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        lmbda,
        epochs,
        eps,
        gamma,
        device,
    )

    _ = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes, seed=seed)
    score = evaluate_policy(env, ppo_agent, n_episodes=20)
    print(f"Expert PPO mean return (20 eps): {score:.2f}")

    n_episode = args.sample_episodes
    expert_s, expert_a = sample_expert_data(env, ppo_agent, n_episode=n_episode, seed=seed)
    total_steps = expert_s.shape[0]

    n_samples = min(args.n_samples, expert_s.shape[0])
    random_index = random.sample(range(expert_s.shape[0]), n_samples)
    expert_s = expert_s[random_index]
    expert_a = expert_a[random_index]

    np.savez(args.output_npz, states=expert_s, actions=expert_a)
    torch.save(ppo_agent.actor.state_dict(), "expert_actor_ppo_cartpole.pth")

    print(f"Sample pool steps before downsampling: {total_steps}")
    print(f"Saved sampled expert pairs: {expert_s.shape[0]}")
    print(f"Saved dataset file: {args.output_npz}")


if __name__ == "__main__":
    main()
