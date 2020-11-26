from typing import List

import gym
import numpy as np
import pyautogui

from actor_critic_with_tensorflow.agent import Agent
from actor_critic_with_tensorflow.utils import plot_learning_curve
from actor_critic_outer_env.environment import WebEnv
from selenium.webdriver import ActionChains
from selenium.webdriver.common import keys
from selenium import webdriver
import time

if __name__ == '__main__':

    print('MAIN')
    # make our environment
    # env = gym.make('CartPole-v0')
    # print('make environment: ', env)
    # change our open ai gym environment on our website
    env = WebEnv()

    # change our inner env action space to our action space
    action_space = env.action_space

    # define our Agent,
    # aplha - learning rate
    # number of actions defined by our environment
    agent = Agent(alpha=1e-5, n_actions=len(action_space))
    print('set agent: alpha=1e-5,  n_actions=', len(action_space))

    # set number of episodes
    n_episodes = 1500
    # print('number of episodes: ', n_episodes)

    # to save plot for checking learning
    filename = 'sberbusiness.png'
    figure_file = fr'plots\sberbusiness.png'

    # to keep track the best score recieved
    # it will default to the lowest range
    # best_score = env.reward_range[0]

    # while we don't know what the env.reward_range[0]
    # best score is we set it to the -1;
    best_score = -1
    # print('best_score: ', best_score)
    # to keep track our score history
    score_history = []

    # boolean to check load or not checkpoints
    load_checkpoint = False

    if load_checkpoint:
        # load models
        agent.load_models()

    # start playing our games
    print('START LEARNING')
    for i in range(n_episodes):
        observation = WebEnv.reset(env)

        # set our terminal flag
        done = False

        # set our score to zero
        score = 0

        # while not done with the episode
        # we can choose action
        while not done:

            coords, action = agent.choose_action(observation)
            # print('action from choose_action: ', action)

            # get the new state, reward, done and info
            # observation_, reward, done, info = env.step(action)
            # print('new observation_: ', observation_)
            # print('reward after action: ', reward)

            # we need to write our own env.step(action) function
            # but it should get coordinates of the action
            counter = 0
            for k in range(len(coords)):
                # TODO: change element to coords
                observation_, reward, done, info, counter = env.step(coords[k], action, counter)
                # increment our score
                score += reward

                # print('score: ', score)
                # if it not load_checkpoint then we want to learn
                if not load_checkpoint:
                   # print('agent learn')
                    agent.learn(observation, reward, observation_, done)

                # either way we want to set the current state to the new state
                # otherwise you will be constantly choosing an action based on
                # initial state of the environment which obviously will not work
                observation = observation_

            # append the score to the store_history to plot the purposes and
            # calculate an average score of previous hundred games
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            # print('average score: ', avg_score)
            # if that average score is better then new best score
            if avg_score > best_score:
                # set the best score to the average score
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()

            print('episode ', i, 'score %.1f ' % score, 'avg_score %.1f' % avg_score)

            # in the end just plot
            # x axis
            if not load_checkpoint:
                x = [i + 1 for i in range(n_episodes)]
                plot_learning_curve(x, score_history, figure_file)
