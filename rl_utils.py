import numpy as np
import torch
from tqdm import tqdm


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(data[: window_size - 1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def reset_env(env, seed=None):
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
        return next_state, reward, done, info
    next_state, reward, done, info = result
    return next_state, reward, done, info


def train_on_policy_agent(env, agent, num_episodes, seed=0):
    return_list = []
    episodes_per_iter = num_episodes // 10
    episode_id = 0

    for i in range(10):
        with tqdm(total=episodes_per_iter, desc=f"Iteration {i}") as pbar:
            for _ in range(episodes_per_iter):
                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                }
                state = reset_env(env, seed=seed + episode_id)
                done = False
                episode_return = 0.0

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = step_env(env, action)

                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)
                episode_id += 1

                if (episode_id) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": f"{episode_id}",
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )
                pbar.update(1)

    return return_list
