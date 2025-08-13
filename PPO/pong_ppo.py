import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
from collections import deque
from torchvision import transforms
from datetime import datetime
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def preprocess(obs):
    return transform(obs).unsqueeze(0).to(device)

# Actor-Critic Network (actor-critic combined structure)
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.policy = nn.Linear(512, action_dim)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.policy(x), self.value(x)

# PPO Agent
class PPOAgent:
    def __init__(self, action_dim):
        self.gamma = 0.99
        self.lr = 0.00025
        self.clip = 0.2
        self.gae_lambda = 0.95
        self.policy = ActorCritic(action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.ppo_epochs = 4
        self.batch_size = 64
        self.entropy_coef = 0.01

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.policy(state)  # logits : unnormalized probability score
            dist = torch.distributions.Categorical(logits=logits)   # logits normalization
            action = dist.sample()
            return action.item(), dist.log_prob(action), dist.entropy()

    # GAE(Generalized Advantage Estimation)
    def compute_gae(self, rewards, values, dones, last_value):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return torch.tensor(returns).float().to(device)
    # policy update
    def update(self, obs, actions, old_log_probs, returns, values):
        advantages = returns - values   # advantage : A(s,a) = Q(s,a) - V(s)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # advantage normalization

        dataset_size = obs.size(0)
        indices = np.arange(dataset_size)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logits, new_values = self.policy(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)  # rt(theta)
                # surr1 : surrogate objective
                surr1 = ratio * batch_advantages
                # surr2 : clipped surrogate objective -> rt(theta) ~ [1-epsilon, 1+epsilon]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages

                policy_loss_actor = -torch.min(surr1, surr2).mean()
                # value loss : new_values(Critic이 예측한 가치함수 값), batch_returns(실제 얻은 return 값)
                value_loss_critic = F.mse_loss(new_values.squeeze(-1), batch_returns)
                loss = policy_loss_actor + 0.5 * value_loss_critic - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

            total_policy_loss += policy_loss_actor.item()
            total_value_loss += value_loss_critic.item()
            total_entropy += entropy.item()

        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }

    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Train PPO
env = gym.make('ALE/Pong-v5', render_mode = 'rgb_array')
action_dim = env.action_space.n
agent = PPOAgent(action_dim)

episodes = 20000
reward_history = []

for episode in range(episodes):
    obs = env.reset()[0]
    obs = preprocess(obs)
    done = False
    total_reward = 0
    # saving experience to trajectory(list)
    trajectory = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'log_probs': [],
        'values': []
    }

    while not done:
        action, log_prob, _ = agent.get_action(obs)
        logits, value = agent.policy(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        next_obs = preprocess(next_obs)

        trajectory['obs'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['dones'].append(done)
        trajectory['log_probs'].append(log_prob.item())
        trajectory['values'].append(value.item())

        obs = next_obs
        total_reward += reward

    with torch.no_grad():
        _, last_value = agent.policy(obs)
    returns = agent.compute_gae(trajectory['rewards'],
                                trajectory['values'],
                                trajectory['dones'],
                                last_value.item())

    obs_tensor = torch.cat(trajectory['obs'])
    actions_tensor = torch.tensor(trajectory['actions']).to(device)
    old_log_probs_tensor = torch.tensor(trajectory['log_probs']).to(device)
    values_tensor = torch.tensor(trajectory['values']).to(device)

    agent.update(obs_tensor, actions_tensor, old_log_probs_tensor, returns, values_tensor)
    reward_history.append(total_reward)

    if episode % 10 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward:.2f}')

env.close()

# Plot
plt.plot(reward_history)
plt.title("PPO on Pong")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.show()

def plot_total_reward(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Total Reward')
    # 100회 에피소드 당 이동 평균 graph
    if len(reward_history) >= 100:
        moving_avg = np.convolve(reward_history, np.ones(100) / 100, mode='valid')
        plt.plot(range(99, len(reward_history)), moving_avg, label='100-episode Moving Average', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid()
    plt.show()

plot_total_reward(reward_history)

# Let agent acts to select greedy actions after finishing training(greedy policy)
env2 = gym.make("ALE/Pong-v5", render_mode='rgb_array')

# 비디오 저장 설정(pygame)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'Pong-v5_{current_time}.avi'
frame_size = (600, 400)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 30.0, frame_size)

agent.epsilon = 0       # greedy policy(probability of random action = 0)
state = env2.reset()[0]
state = preprocess(state)

done = False
total_reward = 0
steps = 0

while not done:
    action, _, _ = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated

    # 현재 frame을 비디오로 저장
    frame = env2.render()

    if frame is not None:
        # numpy 배열로 변환
        frame = np.array(frame)
        if frame.shape[:2] != frame_size[::-1]:
            frame = cv2.resize(frame, frame_size)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow('Pong-v5', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q를 누르면 종료
            break

    next_state = preprocess(next_state)
    state = next_state
    total_reward += reward
    steps += 1
    #env2.render()

    # 진행 상황 출력
    if steps % 10 == 0:
        print(f"Steps: {steps}, Current Reward: {total_reward:.2f}")

print(f"Episode finished.")
print(f"Total Steps: {steps}")
print(f"Total Reward: {total_reward:.2f}")

out.release()
cv2.destroyAllWindows()
env2.close()
print(f"Video has been saved: {video_filename}")