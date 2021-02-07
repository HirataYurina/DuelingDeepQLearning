# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:agent.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras as keras


class DuelingDeepQNetwork(keras.Model):

    def __init__(self, dim1, dim2, num_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = layers.Dense(dim1, activation='relu')
        self.dense2 = layers.Dense(dim2, activation='relu')
        self.V = layers.Dense(1)
        self.A = layers.Dense(num_actions)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class ReplayBuffer:

    def __init__(self, max_size, state_shape):
        self.mem_size = max_size
        self.cntr = 0

        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,), dtype=np.int32)
        self.reward_memory = np.zeros((self.mem_size,), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size,), dtype=np.bool)

    def store_transition(self, state, reward, action, next_state, done):
        index = self.cntr % self.mem_size
        # store transition
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

        self.cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        done = self.done_memory[batch]

        return state, reward, action, next_state, done


class Agent:

    def __init__(self, lr, gamma, epsilon, batch_size,
                 dim1, dim2, num_actions, state_dim,
                 eps_dec=1e-3, eps_end=0.01,
                 max_mem_size=1000000,
                 replace=100):
        self.action_space = [i for i in range(num_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.replace = replace
        self.learning_step = 0

        self.q_val = DuelingDeepQNetwork(dim1, dim2, num_actions)
        self.q_next = DuelingDeepQNetwork(dim1, dim2, num_actions)

        # compile network
        self.q_val.compile(optimizer=Adam(learning_rate=lr),
                           loss='mean_squared_error')
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # init replay buffer
        self.replay_buffer = ReplayBuffer(max_mem_size, state_dim)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, reward, action, next_state, done)

    def get_action(self, observation):

        # epsilon greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            observation = np.array([observation])
            pred = self.q_val.advantage(observation)
            action = tf.argmax(pred, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.replay_buffer.cntr < self.batch_size:
            return

        if self.learning_step % self.replace == 0:
            self.q_next.set_weights(self.q_val.get_weights())

        # start training
        state, reward, action, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)
        q_estimation = self.q_val(state)
        q_target = q_estimation.numpy()
        q_next = self.q_next(next_state)

        max_action = tf.argmax(q_next, axis=1).numpy()

        for i, d in enumerate(done):
            q_target[i, action[i]] = self.gamma * q_next[i, max_action[i]] * (1 - int(d)) + \
                                     reward[i]
        self.q_val.train_on_batch(state, q_target)

        # epsilon descends
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

        self.learning_step += 1
