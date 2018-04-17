# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 10000

class env():
    def __init__(self, stock_data):

        self.stock_data = stock_data
        self.stock_index = 0
        self.last_coin = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str)))
        self.stock_rewards = []
        self.done = False

    def render(self):
        return

    def reset(self):
        self.stock_index = 0
        # TestMatrix = np.reshape(self.stock_data[self.stock_index:8], [64])
        return self.stock_data[self.stock_index]
        # return TestMatrix

    def step(self, action):

        NowData = self.stock_rewards[self.stock_index:]
        # print(NowData)
        gamma = 0.95
        fex = 1 / (0.998 * 0.998) - 1
        f_reward = 0
        for x in range(1,3):
            # print(len(NowData))
            if self.last_coin == action:
                f_reward += gamma * ((NowData[x]-NowData[x-1])/NowData[x-1])
            else:
                f_reward += gamma * (((NowData[x] - NowData[x - 1]) / NowData[x-1])-fex)
            gamma = gamma ** 2
        # action_reward = (NowData*gamma + f_reward)*count
        action_reward  = float(f_reward)
        self.stock_index += 1
        self.last_coin = action
        if self.stock_index >= len(self.stock_data) - 1:
            self.done = True

        return self.stock_data[self.stock_index], action_reward, self.done, 0



class DQNAgent:
    def __init__(self, state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min =  0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        model = Sequential()
        model.add(Dense(self.state_size * 2, input_dim=self.state_size, init='uniform',
                        activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.state_size, activation='relu'))
        model.add(Dense(self.action_size, kernel_initializer='uniform', activation='sigmoid'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def action(self,state):
        act_values = self.model.predict(state)
        self.Q_Value = np.max(act_values)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            # if done:
            #     target[0][action] = reward
            # else:
            a = self.model.predict(next_state)[0]
            t = self.target_model.predict(next_state)[0]
            target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    Data = np.loadtxt(open("./Data/Data.csv", "rb"), delimiter=",", skiprows=0)
    Data = scaler.fit_transform(Data)
    PriceArray = np.loadtxt(open("./Data/PriceArray.csv", "rb"), delimiter=",", skiprows=0)

    env = env(Data)
    state_size = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str))*9)
    action_size = int(state_size/9+1)
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("./DQN_Model/Keras-ddqn.h5")
        print('Loading Keras-ddqn Model Successfully')
    except:
        print('Build New Keras-ddqn Model')
    done = False
    batch_size = 32

    for e in range(EPISODES):

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        Train_Reward = 0
        for time in range(len(Data)-2):
            action = agent.act(state)
            env.stock_rewards = PriceArray[:, action]
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            Train_Reward += reward
            if done:
                agent.update_target_model()
                break
        if e % 1000 == 0:
            print("episode: {}/{}, score:{},e: {:.2}"
                  .format(e, EPISODES, Train_Reward,agent.epsilon))
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 1000 == 0:
            agent.save("./DQN_Model/Keras-ddqn.h5")