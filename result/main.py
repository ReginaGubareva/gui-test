from result.agent import Agent
from result.utils import plot_learning_curve
import numpy as np
import tensorflow as tf
from result.environment import Environment
import random

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    env = Environment()
    score_history = []
    num_episodes = 20
    agent = Agent(alpha=1e-5, n_actions=2)

    filename = 'learn.png'
    figure_file = fr'plots\learn.png'

    best_score = 1
    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    checkpoint_file = open(fr"tmp\trained_model", 'w')
    for i in range(num_episodes):
        counter = 0
        score = 0
        env.reset()
        counter, observation, thresh = env.get_screen(counter)
        contours, centroids = env.get_contours(observation, thresh)
        done = False
        action = agent.choose_action(observation)
        j = len(centroids) - 1
        checkpoint = ''
        # while (not done):
        while j >= 0:
            act = action[centroids[j][0]]
            if act == 0:
                act = 'click'
            else:
                act = 'type'
            checkpoint = checkpoint + str(act) + ' ' + str(centroids[j]) + '; '
            observation_, reward, done, counter = env.step(act, centroids[j], counter)
            score += reward

            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
            j -= 1

        score_history.append(score)

        avg_score = np.mean(score_history[-10:])
        checkpoint = str(checkpoint) + str(avg_score) + '\n'
        print(checkpoint)
        checkpoint_file.write(checkpoint)



        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        # print('episode ', i, 'score %.1f ' % score, 'avg_score %.1f' % avg_score)



    checkpoint_file.close()
    if not load_checkpoint:
        x = [i + 1 for i in range(num_episodes)]
        plot_learning_curve(x, score_history, figure_file)

