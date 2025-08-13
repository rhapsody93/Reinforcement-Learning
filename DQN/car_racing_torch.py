from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
from datetime import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(64 * 8 * 8, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.feature = CNNFeatureExtractor()
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class DQNAgent:
    def __init__(self, device='cpu'):
        self.gamma = 0.99
        self.lr = 0.0005
        self.epsilon = 0.7
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.buffer_size = 100000
        self.batch_size = 128
        self.action_size = 5
        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(self.device)      # original nn
        self.qnet_target = QNet(self.action_size).to(self.device)       # target nn
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # 학습 성능에 따른 적응적 epsilon 조정
    def adaptive_epsilon_adjustment(self, episode_reward):
        # 이동 평균 계산을 위한 reward history
        self.reward_history = getattr(self, 'reward_history', [])
        self.reward_history.append(episode_reward)
        window_size = 10

        # 최근 N개 에피소드의 평균 보상 계산
        recent_rewards = self.reward_history[-window_size:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward

        # 임계값 구간 설정
        very_low = -70
        low = -50
        medium = 100
        high = 300

        # 초기 학습 단계 (처음 200 에피소드)
        early_stage = len(self.reward_history) < 200

        if early_stage:
            # 초기에는 높은 exploration 유지하되 epsilon 선형적으로 감소 0.7 -> 0.5
            self.epsilon = max(0.7 * (1 - len(self.reward_history) / 400), 0.5)
        else:
            if avg_reward < very_low:
                # 매우 낮은 성능
                self.epsilon = max(0.6, self.epsilon * 0.999)
            elif avg_reward < low:
                # 낮은 성능
                self.epsilon = max(0.4, self.epsilon * 0.998)
            elif avg_reward < medium:
                # 중간 성능
                self.epsilon = max(0.2, self.epsilon * 0.997)
            elif avg_reward < high:
                # 좋은 성능
                self.epsilon = max(0.2, self.epsilon * 0.996)

        #avg_reward_threshold = 50  # 임계값

        #if episode_reward < avg_reward_threshold:
            # 성능이 좋지 않으면 탐험 증가
        #    self.epsilon = max(0.8, self.epsilon)
            else:
                # 성능이 좋으면 정상적인 감소 진행
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        qs = self.qnet(state)
        q = qs[torch.arange(self.batch_size), action]

        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.qnet.zero_grad()   # cleargrads
        loss.backward()
        self.optimizer.step()   # update

    def sync_qnet(self):    # syncronization of original & target network
        self.qnet_target.load_state_dict(self.qnet.state_dict())

def preprocess(state):      # def preprocess(state) : global function
    # input image dimension convert : (H, W, C) -> (C, H, W)
    state = np.transpose(state, (2, 0, 1))
    # normalization of image pixel value [0, 255] -> [0, 1]
    state = state / 255.0
    return state

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
episodes = 1000
sync_interval = 20
total_steps = 0
max_steps = 2000000

env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
agent = DQNAgent(device=device)
# torch.save(agent.qnet.state_dict(), 'dqn_qnet_target_1000.pth')
# torch.save(agent.qnet_target.state_dict(), 'dqn_qnet_target_1000.pth')
# agent.qnet.load_state_dict(torch.load('dqn_qnet_target_1000.pth'))
# agent.qnet_target.load_state_dict(torch.load('dqn_qnet_target_1000.pth'))
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    state = preprocess(state)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_state)
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        step_count += 1
        total_steps += 1

        # 최대 step 도달 시 종료
        if total_steps >= max_steps:
            print(f"Reached maximum steps: {max_steps}")
            break

    if total_steps >= max_steps:
        break  # 전체 학습 종료

    if episode % sync_interval == 0:
        agent.sync_qnet()
        # 학습된 모델 저장
        torch.save(agent.qnet.state_dict(), f'dqn_qnet_{episode}.pth')

    # 에피소드가 끝날 때마다 epsilon 조정
    agent.adaptive_epsilon_adjustment(total_reward)

    reward_history.append(total_reward)
    if episode % 10 == 0:
        # print('episode : {}, reward : {}'.format(episode, total_reward))
        print(f'Episode: {episode} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()

# Let agent acts to select greedy actions after finishing training(greedy policy)
#env2 = gym.make('CarRacing-v2', continuous=False, render_mode='human')
env2 = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
# 비디오 저장 설정
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'car_racing_{current_time}.avi'
frame_size = (600, 400)  # CarRacing-v2의 기본 프레임 크기
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 30.0, frame_size)

agent.epsilon = 0       # probability of random action = 0
state = env2.reset()[0]
state = preprocess(state)

done = False
total_reward = 0
steps = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env2.step(action)
    done = terminated | truncated

    # 현재 프레임을 비디오로 저장
    frame = env2.render()

    if frame is not None:
        # numpy 배열로 변환
        frame = np.array(frame)

    #if frame is None or isinstance(frame, (bool, np.bool_)):
    #        continue

    #frame = np.array(frame, dtype=np.uint8)

        if frame.shape[:2] != frame_size[::-1]:
            frame = cv2.resize(frame, frame_size)

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

    # 화면에 표시 (선택사항)
        cv2.imshow('Car Racing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q를 누르면 종료
            break

    next_state = preprocess(next_state)
    state = next_state
    total_reward += reward
    steps += 1
    #env2.render()

    # 진행 상황 출력
    if steps % 100 == 0:
        print(f"Steps: {steps}, Current Reward: {total_reward:.2f}")

print(f"Episode finished.")
print(f"Total Steps: {steps}")
print(f"Total Reward: {total_reward:.2f}")

# 정리
out.release()
cv2.destroyAllWindows()
env2.close()
print(f"Video has been saved: {video_filename}")
