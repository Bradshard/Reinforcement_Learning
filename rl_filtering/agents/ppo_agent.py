import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        value = self.critic(obs)
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_epochs=10):
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist, value = self.policy(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0).detach().numpy(), log_prob.squeeze(0), value.squeeze(0)

    def compute_returns(self, rewards, values, next_value, dones):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, obs, actions, log_probs, returns, values):
        advantages = returns - values
        for _ in range(self.update_epochs):
            dist, value = self.policy(obs)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            ratio = (new_log_probs - log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value.squeeze(-1)).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

