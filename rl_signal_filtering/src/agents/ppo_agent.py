import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(ActorCriticLSTM, self).__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
        lstm_out, _ = self.lstm(obs)
        out = lstm_out[:, -1, :]
        mean = self.actor_head(out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(obs[:, -1, :])
        return dist, value

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_epochs=10):
        self.policy = ActorCriticLSTM(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,D]
        dist, value = self.policy(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.update_epochs):
            dist, value = self.policy(obs.unsqueeze(1))  # add sequence dimension
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value.squeeze(-1)).pow(2).mean()
            entropy_bonus = dist.entropy().sum(dim=-1).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
