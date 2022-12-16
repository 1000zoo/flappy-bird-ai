import sys
import flappy_bird_gym
import pylab
import random
import time
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from matplotlib import cm

# for cnn
from PIL import Image
import os

from os.path import join

"""
CNN 을 이용한 DQN
연속된 4장이 아닌, 한 장의 사진을 학습
reward shaping => 라이브러리 내부에서 구현
"""

IMAGE_SIZE = (128, 128, 1)
TRIAL = "_1"

MODEL_NAME = os.path.basename(__file__).split('.')[0] + TRIAL
FIGURE_PATH = os.path.join('save_graph', MODEL_NAME + '.png')
MODEL_PATH = os.path.join('save_model', MODEL_NAME + '.h5')


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        # self.load_model = True

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.0001
        self.batch_size = 32
        self.train_start = 5000

        # 리플레이 메모리, 최대 크기 50000
        self.memory = deque(maxlen=50000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())

        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=10.))
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
        states = np.array([sample[0][0] / 255. for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3][0] / 255. for sample in mini_batch], dtype=np.float32)
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
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


def pre_processing(o):
    ## s에는 bytes array 로 현재 state 의 image (점수는 표시되지 않음)
    ## RGBA 이미지를 흑백화면으로 바꾼 후 불필요한 바닥 부분을 줄임
    ## crop(left, up, right, down)
    img = Image.frombuffer('RGBA', (288, 512), o) \
        .crop((0, 0, 288, 425)) \
        .convert("L") \
        .resize(IMAGE_SIZE[:2])
    return np.asarray(img).reshape([1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])


if __name__ == "__main__":
    EPISODES = 1000

    ## cnn=True => step 의 리턴값이 dx, dy 에서 이미지 버퍼로 바뀜
    ## background='black' => 불필요한 배경을 학습하지 않도록 검은색 배경으로 변경
    env = flappy_bird_gym.make('FlappyBird-v0', cnn=True, background='black')
    state_size = IMAGE_SIZE
    action_size = env.action_space.n
    agent = DQNAgent(action_size=action_size, state_size=state_size)
    max_score = 0
    scores, episodes = [], []

    for e in range(EPISODES + 1):
        done = False

        env.reset()
        env.render()
        observe = env.step(0)[0]
        score = 0
        state = pre_processing(observe)

        while not done:
            env.render()

            action = agent.get_action(state)
            observe, reward, done, info = env.step(action)
            next_state = pre_processing(observe)
            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score = _get_score(info)
            state = next_state

            if done:
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
