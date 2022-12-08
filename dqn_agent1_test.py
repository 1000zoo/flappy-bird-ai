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
from tensorflow.keras import models

from os.path import join

"""
TEST
DQN 에서 exploration 을 제거한 뒤 학습 중 보다는 스코어가 꽤 높았다.
"""

MODEL_NAME = "dqn_agent_21_best_until_ep992"
FIG_DIR = "./save_graph"
MODEL_DIR = "./save_model"

FIGURE_PATH = join("save_graph",  MODEL_NAME + ".png")
MODEL_PATH = join(MODEL_DIR, MODEL_NAME + ".h5")


# Mountain-car 예제에서의 DQN 에이전트를 차용
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32
        self.train_start = 3000
        self.nn_size = 30

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.model.load_weights(MODEL_PATH)

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

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        q_value = self.model.predict(state, verbose=0)
        return np.argmax(q_value[0])


def _get_score(info: dict) -> int:
    return info['score']


if __name__ == "__main__":
    EPISODES = 10
    env = flappy_bird_gym.make('FlappyBird-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.reset()
    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

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

            reward -= pow(state[0][1] * state[0][0], 2)

            # 매 타임스텝마다 학습

            if _get_score(info) > score:
                print('next_state:', next_state)

            score = _get_score(info)
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score)

        pylab.plot(episodes, scores, 'b')
        pylab.savefig(FIGURE_PATH)
