from typing import List

import gym
import numpy as np
from actor_critic_with_tensorflow.agent import Agent
from actor_critic_with_tensorflow.utils import plot_learning_curve

if __name__ == '__main__':

    # make our environment
    env = gym.make('CartPole-v0')

    # print('make environment: ', env)
    # define our Agent,
    # aplha - learning rate
    # number of actions defined by our environment
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)

    # set number of games
    n_games = 1500
    # print('number of games: ', n_games)
    filename = 'cartpole.png'
    figure_file = fr'plots\cartpole.png'

    # to keep track the best score recieved
    # it will default to the lowest range
    best_score = env.reward_range[0]
    # print('best_score: ', env.reward_range[0])
    # to keep track our score history
    score_history = []

    # boolean to check load or not checkpoints
    load_checkpoint = False

    if load_checkpoint:
        # load models
        agent.load_models()

    # start playing our games
    for i in range(n_games):
        # reset our environment
        observation = env.reset()
        # print('observation: ', observation)

        # set our terminal flag
        done = False

        # set our score to zero
        score = 0

        # while not done with the episode
        # we can choose action
        while not done:
            action = agent.choose_action(observation)

            # get the new state, reward, done and info
            observation_, reward, done, info = env.step(action)

            # increment our score
            score += reward
       
            # if it not load_checkpoint then we want to learn
            if not load_checkpoint:
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
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

