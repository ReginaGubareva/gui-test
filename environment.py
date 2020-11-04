import random

import gym
import numpy as np
from gym import spaces
from gym.utils import closer

env_closer = closer.Closer()


class AuthEnvironment(gym.Env):

    # I don't understand why we should write it here,
    # but we should because some causes
    metadata = {'render_modes': ['human']}
    spec = None

    # We should define the possible actions
    # Define it as constants
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    CLICK = 4
    TYPE_CREDENTIALS = 5

    # Count of actions
    N_DISCRETE_ACTIONS = 6

    HEIGHT = 128
    WIDTH = 128
    N_CHANNELS = 0
    LOGIN = 'sber5'
    PASSWORD = 'Qwert123!'

    # Define the minimal and maximal reward for the agents
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, current_state):

        super(AuthEnvironment, self).__init__()

        # Define actions and observation space
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)

        # Our observation is an 128x128 RGB(means 3) image:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.current_state = current_state

    def reset(self):
        state = self.current_state
        return state

    def render(self, mode='human'):
        pass

    def step(self, action):
        return

