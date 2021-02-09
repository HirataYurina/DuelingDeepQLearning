# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:agent_play.py
# software: PyCharm

from dueling_dqn.agent import DuelingDeepQNetwork
import gym
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# get network
dueling_dqn = DuelingDeepQNetwork(dim1=128, dim2=128, num_actions=4)
dummy = keras.Input(shape=(8,))
dueling_dqn(dummy)
dueling_dqn.load_weights('./lunarlander-2.h5')

env = gym.make(id='LunarLander-v2')

# reset
observation = np.array([env.reset()])

done = False

total_reward = 0
while not done:
    action = tf.argmax(dueling_dqn(observation), axis=1).numpy()[0]
    observation, reward, done, info = env.step(action)
    observation = np.array([observation])
    env.render()
    total_reward += reward
print(total_reward)
