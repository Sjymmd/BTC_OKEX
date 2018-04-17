# coding: utf-8
from __future__ import division
import tensorflow as tf
import numpy as np
import random
from collections import deque
from fractions import Fraction
import warnings

warnings.filterwarnings("ignore")

class TWStock():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.stock_index = 0
        self.last_coin = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str)))
        self.stock_rewards = []

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
            stock_done = True
        else:
            stock_done = False
        return self.stock_data[self.stock_index], action_reward, stock_done, 0

def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
# INITIAL_EPSILON = 0.5  # starting value of epsilon
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 3000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
memory_size=500
batch_size=64
learning_rate=0.001
e_greedy_increment=0.0001
e_greedy=0.9

class DQN():
    # DQN Agent
    def __init__(self,self_print = True):
        self.print = self_print
        self.memory_size = 500
        self.dueling = True
        # init experience replay
        self.replay_buffer = deque()
        self.lr = learning_rate
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.gamma = GAMMA
        self.epsilon_increment = e_greedy_increment
        self.epsilon_learn = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon_max = e_greedy
        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.state_dim = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str))*9)
        self.action_dim =int(self.state_dim/9+1)

        self.n_features = int(len(np.loadtxt("./Log/Coin_Select.txt", dtype=np.str))*9)
        self.n_actions = int(self.state_dim/9+1)
        self.cost_his = []

        self.create_Q_network()
        self.create_training_method()

        self.replace_target_iter = 50
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        checkpoint = tf.train.get_checkpoint_state("DQN_Model")

        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            if self.print is True:
                print(
                "Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_Q_network(self):

        self.state_input = tf.placeholder("float", [None, self.state_dim],name='s')
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1],initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1],initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
            return out

            # ------------------ build evaluate_net ------------------
        # self.s = tf.placeholder("float", [None, self.state_dim])  # input
        self.q_target = tf.placeholder("float", [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.n_features, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.state_input, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder("float", [None, self.state_dim],name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.state_input, c_names, n_l1, w_initializer, b_initializer)


    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply (self.q_eval, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.cost)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.y_input, 1), tf.argmax(self.y_input, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy = accuracy
        tf.summary.scalar('accuracy', accuracy)
        # print('accuracy',accuracy)
        tf.summary.scalar("cost", self.cost)
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
        Q_value_batch = self.q_eval.eval(feed_dict={self.state_input: next_state_batch})

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
            self.saver.save(self.session, './DQN_Model/' + 'network' + '-dqn', global_step=self.time_step)

    def egreedy_action(self, state):
        Q_value = self.q_eval.eval(feed_dict={
            self.state_input: [state]})[0]

        self.Q_Value = np.amax(Q_value)

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100000

    def action(self, state):
        self.Q_Value = np.amax(self.q_eval.eval(feed_dict={
            self.state_input: [state]})[0])
        return np.argmax(self.q_eval.eval(feed_dict={
            self.state_input: [state]})[0])


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.session.run(self.q_next, feed_dict={self.state_input: batch_memory[:, -self.n_features:]}) # next observation
        q_eval = self.session.run(self.q_eval, {self.state_input: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.session.run([self._train_op, self.loss],
                                     feed_dict={self.state_input: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon_learn = self.epsilon_learn + self.epsilon_increment if self.epsilon_learn < self.epsilon_max else self.epsilon_max
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        self.learn_step_counter += 1
        return self.cost
