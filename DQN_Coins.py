# coding: utf-8
from __future__ import division

import tensorflow as tf
import numpy as np
import random
from collections import deque
import datetime
import matplotlib.pyplot as plt
import cv2
import time
import matplotlib.image as pimg

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

from Volume_Early_Warning import *
import pandas as pd
from OkcoinSpotAPI import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
Okex_Api = Okex_Api()
Coin = Okex_Api.GetCoin()
# Coin = ['snt_usdt']
Okex_Api._CoinLenth = len(Coin)
Okex_Api._KlineChosen = '1hour'
Okex_Api._Lenth = 24*100
Okex_Api._EndLenth = 0
# now = datetime.datetime.now()
# now = now.strftime('%Y-%m-%d %H:%M:%S')
# print(now)
StartTime = time.time()

def Get_Dataframe(Coin):
    try:
        DataFrame = pd.DataFrame(columns=("Coin", "Cny","High","Low", "Inc", "Volume_Pre_K", "Mean_Volume_K", "_VolumeS", "_VolumeM"))
        data = pd.DataFrame(okcoinSpot.getKline(Okex_Api._Kline[Okex_Api._KlineChosen], Okex_Api._Lenth, Okex_Api._EndLenth, x)).iloc[:Okex_Api._Lenth-1, ]
        data[5] = data.iloc[:, 5].apply(pd.to_numeric)
        data = data[data[5] >= 1000]
        data = data.reset_index(drop=True)
        Increase = (float(data.iloc[0, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
        # Increase = str('%.2f' % (Increase) + '%')
        price = float(data.iloc[0, 4])
        Hi_price = round(float((data.iloc[0, 2]))* Okex_Api._USDT_CNY,2)
        Lo_price = round(float((data.iloc[0, 3]))* Okex_Api._USDT_CNY,2)
        Cny = round(price * Okex_Api._USDT_CNY, 2)
        Volume = float(data.iloc[0, 5])
        Volume_Mean = round(Volume/1000,2)
        Volume_Pre = round(Volume / 1000, 2)
        Volume_Pre_P = 0
        if Volume_Mean == 0:
            Volume_Inc = 0
        else:
            Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
        Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny,'High':Hi_price,'Low':Lo_price, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                               'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
        DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
        for lenth in range(1,Okex_Api._Lenth-1):
            try:
                Increase = (float(data.iloc[lenth, 4]) - float(data.iloc[0, 1])) / float(data.iloc[0, 1]) * 100
                # Increase = str('%.2f' % (Increase) + '%')
                price = float(data.iloc[lenth, 4])
                Hi_price = round(float((data.iloc[lenth, 2])) * Okex_Api._USDT_CNY, 2)
                Lo_price = round(float((data.iloc[lenth, 3])) * Okex_Api._USDT_CNY, 2)
                Cny = round(price * Okex_Api._USDT_CNY, 2)
                Volume = data.iloc[:lenth+1, 5].apply(pd.to_numeric)
                Volume_Mean = round(Volume.mean() / 1000, 2)
                Volume_Pre = round(Volume.iloc[lenth] / 1000, 2)
                Volume_Pre_P = round((Volume[lenth] / Volume[lenth - 1])-1, 2)
                Volume_Inc = round(((Volume_Pre - Volume_Mean) / Volume_Mean), 2)
                Timeshrft = pd.Series({'Coin': Coin, 'Cny': Cny,'High':Hi_price,'Low':Lo_price, 'Inc': Increase, 'Volume_Pre_K': Volume_Pre,
                                       'Mean_Volume_K': Volume_Mean, '_VolumeS': Volume_Pre_P, '_VolumeM': Volume_Inc})
                DataFrame = DataFrame.append(Timeshrft, ignore_index=True)
            except:
                break
        return DataFrame
        # print(DataFrame)
    except:
        print('%sError'%Coin)

class TWStock():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_index = 0

    def render(self):
        # 尚未實作
        return

    def reset(self):
        self.stock_index = 0
        return self.stock_data[self.stock_index]

    # 0: 觀望, 1: 持有多單, 2: 持有空單
    def step(self, action):
        self.stock_index += 1
        my_train = self.stock_data[self.stock_index:]
        max = np.amax(my_train, axis=0)[1]
        action_reward = max - self.stock_data[self.stock_index][0]
        # action_reward = self.label[self.stock_index]
        if (action == 0):
            action_reward = 0

        if (action == 2):
            action_reward = -1 * action_reward
        # print(str(action)+" "+str(action_reward))

        # cv2.imshow('image_mean2', self.stock_data[self.stock_index])
        # cv2.waitKey(1)

        stock_done = False
        if self.stock_index >= len(self.stock_data) - 1:
            stock_done = True
        else:
            stock_done = False
        return self.stock_data[self.stock_index], action_reward, stock_done, 0

        # self.stock_data[self.stock_index]

def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env):

        self.dueling = True
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON

        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n

        self.state_dim = 8
        self.action_dim = 3

        self.create_Q_network()
        self.create_training_method()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print(
            "Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print
            ("Could not find old network weights")

    def create_Q_network(self):
        # -----------------------end cnn   start

        W1 = self.weight_variable([self.state_dim, 8])
        b1 = self.bias_variable([8])
        W2 = self.weight_variable([8, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2


    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply (self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)

        # tf.scalar_summary("cost", values=self.cost)
        # tf.histogram_summary("cost", values=self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        # print(reward_batch)
        next_state_batch = [data[3] for data in minibatch]
        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        # save network every 100 iteration
        if self.time_step % 50 == 0:
            self.saver.save(self.session, 'Saved_networks/' + 'network' + '-dqn', global_step=self.time_step)

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]})[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------


# ## main function


# Hyper Parameters
EPISODE = 200  # Episode limitation
# 300 # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)

    env = TWStock(my_train)
    agent = DQN(env)

    print('Start')
    train_output = ""
    rate_string = ""
    for episode in range(EPISODE):

        # initialize task
        state = env.reset()

        # Train
        out = "train\n"
        train_reward = 0
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for trai
            next_state, reward, done, _ = env.step(action)
            out += str(reward) + " "
            train_reward += reward
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        anal = out.split()
        p = 0.0
        n = 0.0
        for x in range(1, len(anal) - 1):
            if (float(anal[x]) > 0):
                p += float(anal[x])
            elif (float(anal[x]) < 0):
                n += float(anal[x])

        rate = round(p / (n * (-1) + p), 2)
        rate_string += str(rate) + " "
        # fo.write(out + "\n")
        train_output += str(train_reward) + " "
        # Test every 100 episodes
        if episode % 10 == 0:
            out = "test\n"
            env1 = TWStock(my_test)
            total_reward = 0

            for i in range(TEST):
                state = env1.reset()

                for j in range(STEP):
                    env1.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env1.step(action)
                    out += str(action) + " " + str(reward) + ","
                    total_reward += reward
                    if done:
                        break
            # fo.write(out + "\n")
            ave_reward = total_reward / TEST
            print(train_output)
            train_output = ""
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward, 'training Rate past10:', rate_string)
            rate_string = ""
            # if ave_reward >= 1000:
            #     print('End')
            #     break



if __name__ == '__main__':

    # Coin = ['snt_usdt']
    for x in Coin[:int(Okex_Api._CoinLenth)]:

        try:
            DataFrame = Get_Dataframe(x)
            Data = DataFrame.iloc[:, 1:]
            lenth = int(len(Data) * 5 / 6)
            STEP = lenth - 1
            Train = Data.iloc[:lenth, ]
            my_train = scaler.fit_transform(Train)
            Test = Data.iloc[lenth:, ]
            my_test = scaler.fit_transform(Test)
            # y = open("./TXT/%s.txt"%x, "w")
            # y.close()
            # fo = open("./TXT/%s.txt"%x, "a")
            main()
        except:
            continue

    Coin = ['btc_usdt']
    scaler = preprocessing.StandardScaler()
    scaler_Price = preprocessing.StandardScaler()
    for x in Coin:
        TestData = Get_Dataframe(Coin)
        TestData = TestData.iloc[:, 1:]

        TestPrice = TestData.iloc[:, 0]
        TestPrice = TestPrice.reshape(-1,1)
        TestPrice = scaler_Price.fit_transform(TestPrice)

        TestData_Initial = TestData.as_matrix()
        TestData = scaler.fit_transform(TestData_Initial)

        env1 = TWStock(TestData)
        state = env1.reset()
        agent = DQN(env1)
        Cny = 1000
        Coin = 0
        for i in range(len(TestData)-1):
            env1.render()
            action = agent.action(state)  # direct action for test
            state, reward, done, _ = env1.step(action)
            Price = scaler_Price.inverse_transform(state[0].reshape(-1,1))
            Price = round(Price[0][0],2)
            if action == 1 and done is False and Cny>0:
                print('Buy',i,Price,Cny +Coin*Price-1000)
                Coin = Cny/Price
                Cny = 0
            elif action ==2 and Coin >0 and done is False:
                Cny = Coin * Price
                Coin = 0
                print('Sell', i, Price,Cny +Coin*Price-1000)
        profit = Cny +Coin*Price-1000
        print('Profit:%d'%profit)
