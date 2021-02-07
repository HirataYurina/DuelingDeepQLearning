# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm

from dueling_dqn.agent import Agent
import gym


if __name__ == '__main__':
    game_env = gym.make('LunarLander-v2')
    agent = Agent(lr=0.0005, gamma=0.99, epsilon=1.0, batch_size=64,
                  dim1=128, dim2=128, num_actions=4, state_dim=[8])

    n_game = 300
    scores = []

    for i in range(n_game):
        done = False
        score = 0
        observation = game_env.reset()
        while not done:
            action = agent.get_action(observation)
            next_state, reward, done, _ = game_env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, next_state, done)
            observation = next_state
            agent.learn()
        scores.append(score)
        print('episode', i, 'epsilon %.3f' % agent.epsilon, 'score %.1f' % score)

    # save weights
    q_val = agent.q_val
    q_val.save_weights('lunarlander-2.h5')
