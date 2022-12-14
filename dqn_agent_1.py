import sys
import flappy_bird_gym
import pylab
import random
import time
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

import os

"""
DQN
Replay memory 의 최대를 30000 / start 3000
바로 다음의 state 만 사용
reward shaping =>   state 로 agent 의 위치와 파이프 중간 지점 사이의 x, y 좌표 차이값이 주어지는데
                    이 차이값의 곱의 제곱을 reward (매 step 마다 1 씩 주어짐) 에서 뺌.
"""

### TODO
### 실행마다 Trial 변경해주기


TRIAL = "1"

# MODEL_NAME = os.path.basename(__file__).split('.')[0] + TRIAL
MODEL_NAME = "dqn_agent_wo_rs" + TRIAL
FIGURE_PATH = os.path.join('save_graph', MODEL_NAME + '.png')
MODEL_PATH = os.path.join('save_model', MODEL_NAME + '.h5')


# Mountain-car 예제에서의 DQN 에이전트를 차용
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.batch_size = 32
        self.train_start = 3000
        self.nn_size = 30

        # 리플레이 메모리, 최대 크기 20000
        self.memory = deque(maxlen=30000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()

        model.add(Dense(self.nn_size, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.nn_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state, verbose=0)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


def _get_score(info: dict) -> int:
    return info['score']


def _get_model_path(ep):
    file = MODEL_NAME + "_best_until_ep" + str(ep) + '.h5'
    return os.path.join('save_model', file)


if __name__ == "__main__":
    EPISODES = 300
    env = flappy_bird_gym.make('FlappyBird-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.reset()
    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    max_score = 0

    for e in range(EPISODES + 1):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # if state[0][0] < 0.53 and state[0][1] > 0.4 or state[0][1] < -0.5:
            #     done = True
            #     reward -= 15

            # reward -= pow(state[0][1] * state[0][0], 2)
            # print(reward)

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score = _get_score(info)
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                # 최고 기록을 갱신할 때 마다 새로운 이름으로 모델 저장
                if max_score < score:
                    agent.model.save_weights(_get_model_path(e))
                    max_score = score

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

        if e % 10 == 0:
            agent.model.save_weights(MODEL_PATH)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig(FIGURE_PATH)
