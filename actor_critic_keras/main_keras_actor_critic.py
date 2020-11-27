import gym
import os
from actor_critic_keras import Agent
from utils import plotLearning
from gym import wrappers
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    #env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v0')
    score_history = []
    num_episodes = 6 #000
    agent = Agent(alpha=0.00001, beta=0.00005, n_actions=env.action_space.n,
                  input_dims=env.observation_space.shape[0])

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i,'score: %.2f' % score,
              'avg score %.2f' % avg_score)

    filename = 'LunarLander.png'
    plotLearning(score_history, filename=filename, window=100)