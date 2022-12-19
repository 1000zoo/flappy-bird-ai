import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as ppl

import os


# Mountain-car 예제에서의 DQN 에이전트를 차용
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.999
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.00001
        self.batch_size = 32
        self.train_start = 3000
        self.nn_size = 60

        # 리플레이 메모리, 최대 크기 60000
        self.memory = deque(maxlen=60000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        _model = Sequential()

        _model.add(Dense(self.nn_size, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        _model.add(Dense(self.nn_size, activation='relu', kernel_initializer='he_uniform'))
        _model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        _model.summary()
        _model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return _model

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        q_value = self.model.predict(state, verbose=0)
        return np.argmax(q_value[0])


if __name__ == "__main__":
    XMIN = 0
    XMAX = 0.5
    YMIN = -0.5
    YMAX = 0.5
    PATH = "model_for_pt/"
    # DQN 에이전트 생성
    agent = DQNAgent(2, 2)

    scores, episodes = [], []
    max_score = 0

    pos_x = np.arange(XMIN, XMAX, step=0.02)
    pos_y = np.arange(YMIN, YMAX, step=0.02)

    color = ['red', 'blue']  # red : 낙하 / blue : 점프
    models = os.listdir(PATH)

    for model in models:
        for dx in pos_x:
            for dy in pos_y:
                temp = np.array((dx, dy))
                temp = np.reshape(temp, [1, 2])
                ppl.scatter(dx, dy, c=color[agent.get_action(temp)])

        _name = model.split(".")[0]
        ppl.xlabel("dx")
        ppl.ylabel("dy")
        ppl.title(_name)
        ppl.savefig("policy_table_by_" + _name + ".png")
        ppl.close()

