import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agents.ppo_agent import PPOAgent
from envs.signal_env import SignalFilterEnv

# Hyperparameters
NUM_EPISODES = 500
EPISODE_LENGTH = 1
FILTER_ORDER = 5
SIGNAL_LENGTH = 1024
GAMMA = 0.99
UPDATE_EPOCHS = 10
LR = 3e-4
CLIP_EPS = 0.2
MODEL_PATH = "ppo_filter.pt"

# Environment setup
env = SignalFilterEnv(signal_length=SIGNAL_LENGTH, filter_order=FILTER_ORDER)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Agent setup
agent = PPOAgent(obs_dim, act_dim, lr=LR, gamma=GAMMA, clip_eps=CLIP_EPS, update_epochs=UPDATE_EPOCHS)

avg_rewards = []
ppo_losses = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()

    obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []

    for t in range(EPISODE_LENGTH):
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

    avg_reward = np.mean(rew_list)
    avg_rewards.append(avg_reward)
    ppo_losses.append(loss.item())

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}: Avg Reward = {avg_reward:.4f}")

    if (episode + 1) % 50 == 0:
        torch.save(agent.policy.state_dict(), MODEL_PATH)
        print(f"Model saved at episode {episode+1}")

# Plot reward and loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(avg_rewards, label="Avg Reward")
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ppo_losses, label="Loss", color='orange')
plt.title("PPO Loss per Update")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.show()
