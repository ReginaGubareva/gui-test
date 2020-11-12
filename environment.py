import gym
from PIL import Image
from selenium import webdriver
from gym import spaces
import pyautogui
import time

from selenium.webdriver import ActionChains
from selenium.webdriver.common import keys


class Environment:
    def __init__(self):
        self.driver = webdriver.Chrome(fr'D:\chromedriver.exe')
        self.action_space = ['click', 'type']

        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        # (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        # Вроде как у меня не должно быть observation space

    def step(self, action, coordinates):
        reward = 0
        done = False
        actions = ActionChains(self.driver)

        actions.move_to_element_with_offset(self.driver.find_element_by_tag_name('body'), 0, 0)
        if action == 'click':
            actions.move_by_offset(coordinates[0], coordinates[1]).click()
            s_t = pyautogui.screenshot()
            if (s_t.isTerminal()):
                # reward += 1
                done = True
            else:
                reward -= 1
        if action == 'type':
            actions.move_by_offset(coordinates[0], coordinates[1]).send_keys(keys)
            s_t = pyautogui.screenshot()
            if (not self.isTerminal(s_t) and self.haveCredentials(s_t)):
                reward += 1
            else:
                reward -= 1

    def reset(self):
        self.driver.get("https://www.google.com/?hl=en")
        time.sleep(3)
        state = pyautogui.screenshot()

        return state

    def isTerminal(self, state):
        terminal_state_1 = Image.open(r'D:\gui-test\resources\terminal_state_1.PNG')
        terminal_state_2 = Image.open(r'D:\gui-test\resources\start_state.png')
        if(open(state, "rb").read() == open(terminal_state_1,"rb").read()
                or open(state, "rb").read() != open(terminal_state_2, "rb").read()):
            return True
        else:
            return False

    def haveCredentials(s_t):

        return True