from result.agent import Agent
from result.utils import plot_learning_curve
import numpy as np
import tensorflow as tf
from result.environment import Environment

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    env = Environment()
    score_history = []
    num_episodes = 256
    agent = Agent(alpha=1e-5, n_actions=2)

    filename = 'learn.png'
    figure_file = fr'plots\learn.png'

    x = []
    y = []
    n = 256 * 256

    for i in range(256):
        x.append(i)
        y.append(i)

    c = [[x0, y0] for x0 in x for y0 in y]

    best_score = 1
    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(num_episodes):
        counter = 0
        env.reset()
        counter, observation = env.get_screen(counter)
        done = False
        score = 0
        while not done:
            for j in range(256):
                action = agent.choose_action(observation)
                observation_, reward, done, counter = env.step(action, c[j], counter)
                score += reward
                if not load_checkpoint:
                    agent.learn(observation, reward, observation_, done)
                observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f ' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(num_episodes)]
        plot_learning_curve(x, score_history, figure_file)

