# coding: utf-8
from __future__ import division
import tensorflow as tf
import numpy as np
import random
from collections import deque
import pandas as pd
from fractions import Fraction
import warnings

warnings.filterwarnings("ignore")

class TWStock():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_index = 0

    def render(self):
        return

    def reset(self):
        self.stock_index = 0
        # TestMatrix = np.reshape(self.stock_data[self.stock_index:8], [64])
        return self.stock_data[self.stock_index]
        # return TestMatrix

    def step(self, action):
        self.stock_index += 1
        if action % 3 == 0:
            action_reward = 0
        if action % 3 == 1:
            my_train = self.stock_data[self.stock_index:]
            num = int(Fraction(8, 3)*action- Fraction(5, 3))
            max = np.amax(my_train, axis=0)[num]
            action_reward = max - self.stock_data[self.stock_index][num-1]
        if action % 3 == 2:
            my_train = self.stock_data[self.stock_index:]
            num = int(Fraction(8, 3)*(action-1)- Fraction(5, 3))
            max = np.amax(my_train, axis=0)[num]
            action_reward = max - self.stock_data[self.stock_index][num - 1]
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

        self.state_dim = int(len(pd.read_table('Coin_Select.txt', sep=',').iloc[:5, 0].values)*8)
        self.action_dim =int(self.state_dim/8*3)

        self.create_Q_network()
        self.create_training_method()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("Save_Dueling_Networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print(
            "Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print
            ("Could not find old network weights")

    def create_Q_network(self):
        # -----------------------end cnn   start

        W1 = self.weight_variable([self.state_dim, self.state_dim])
        b1 = self.bias_variable([self.state_dim])
        W2 = self.weight_variable([self.state_dim, self.action_dim])
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
            self.saver.save(self.session, 'Save_Dueling_Networks/' + 'network' + '-dqn', global_step=self.time_step)

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