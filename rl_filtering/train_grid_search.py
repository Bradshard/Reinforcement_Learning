import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agents.ppo_agent import PPOAgent
from envs.signal_env import SignalFilterEnv
from itertools import product
import os
from multiprocessing import Pool, cpu_count
import pickle

# Expanded search grids
LR_GRID = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
CLIP_EPS_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
GAMMA_GRID = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
UPDATE_EPOCHS_GRID = [3, 5, 10, 15, 20]
FILTER_ORDER_GRID = [2, 3, 5, 7, 9, 12]
EPISODE_LENGTH_GRID = [1, 5, 10]
NUM_EPISODES_GRID = [100, 200]
SIGNAL_LENGTH = 1024

os.makedirs("grid_results", exist_ok=True)

def run_config(config):
    lr, clip_eps, gamma, update_epochs, filter_order, episode_length, num_episodes = config
    tag = f"lr{lr}_clip{clip_eps}_gamma{gamma}_upd{update_epochs}_fo{filter_order}_ep{episode_length}_ne{num_episodes}"
    print(f"Running {tag}")

    env = SignalFilterEnv(signal_length=SIGNAL_LENGTH, filter_order=filter_order)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, lr=lr, gamma=gamma, clip_eps=clip_eps, update_epochs=update_epochs)

    avg_rewards, ppo_losses = [], []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

        for _ in range(episode_length):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(torch.tensor(obs, dtype=torch.float32))
            act_list.append(torch.tensor(action, dtype=torch.float32))
            logp_list.append(log_prob.detach())
            val_list.append(value.detach())
            rew_list.append(reward)
            done_list.append(terminated)

            obs = next_obs
            if terminated or truncated:
                break

        with torch.no_grad():
            _, next_value = agent.policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            next_value = next_value.item()

        returns = agent.compute_returns(rew_list, val_list, next_value, done_list)

        obs_tensor = torch.stack(obs_list)
        act_tensor = torch.stack(act_list)
        logp_tensor = torch.stack(logp_list)
        val_tensor = torch.stack(val_list)
        advantages = returns - val_tensor

        for _ in range(agent.update_epochs):
            dist, value = agent.policy(obs_tensor)
            new_log_probs = dist.log_prob(act_tensor).sum(axis=-1)
            ratio = (new_log_probs - logp_tensor).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - agent.clip_eps, 1 + agent.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value.squeeze(-1)).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        avg_rewards.append(np.mean(rew_list))
        ppo_losses.append(loss.item())

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, label="Avg Reward")
    plt.title("Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ppo_losses, label="Loss", color='orange')
    plt.title("PPO Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"grid_results/{tag}.png")
    plt.close()

    torch.save(agent.policy.state_dict(), f"grid_results/{tag}.pt")

    return (tag, avg_rewards[-1])

if __name__ == "__main__":
    configs = list(product(
        LR_GRID,
        CLIP_EPS_GRID,
        GAMMA_GRID,
        UPDATE_EPOCHS_GRID,
        FILTER_ORDER_GRID,
        EPISODE_LENGTH_GRID,
        NUM_EPISODES_GRID,
    ))

    with Pool(cpu_count()) as pool:
        results = pool.map(run_config, configs)

    best_config = max(results, key=lambda x: x[1])
    with open("grid_results/grid_log.txt", "w") as f:
        for tag, final_reward in results:
            f.write(f"{tag} - Final Reward: {final_reward:.4f}\n")
        f.write(f"\nBest Config: {best_config[0]} with Final Reward: {best_config[1]:.4f}\n")

    os.rename(f"grid_results/{best_config[0]}.pt", "ppo_filter_best.pt")
    print(f"Best model saved as ppo_filter_best.pt")
