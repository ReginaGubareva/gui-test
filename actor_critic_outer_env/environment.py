import pyautogui
from PIL import Image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common import keys
import cv2
import time
import numpy as np

class WebEnv:
    def __init__(self):
        self.action_space = ['click', 'type']
        self.driver = webdriver.Chrome(fr'D:\chromedriver.exe')
        self.driver.get('https://digital.sberbank.kz/customer/login')
        time.sleep(5)

    # this function reset environment to the initial state
    # then we just save our initial state and return it
    def reset(self):
        state = Image.open(fr'D:\gui-test\resources\initial.png')
        return state

    def get_screen(self, counter):
        element = self.driver.find_element_by_class_name('loginWindow')
        filename = "%d" % counter
        state = element.screenshot(fr'resources\learning_screens\{filename}.png')
        # state = element.screenshot_as_png
        image = Image.open(fr'D:\gui-test\resources\learning_screens\{filename}.png')
        new_image = image.resize((256, 256))
        greyscale_image = new_image.convert('L')
        greyscale_image.save(fr'resources\learning_screens\{filename}.png')
        counter += 1
        return greyscale_image, counter


    def isTerminal(self, state):
        terminal_state = cv2.imread(fr'D:\gui-test\resources\terminal_state.png')
        # screen = cv2.imread(fr'D:\gui-test\resources\learning_screens\2.png')
        # in terminal state we have credentials and the sign with wrong credentials
        difference = cv2.subtract(terminal_state, state)
        result = not np.any(difference)
        if result is True:
            print("Pictures are the same")
            return True
        else:
            cv2.imwrite(fr'resources\ed.jpg', difference )
            print("Pictures are different, the difference is stored as ed.jpg")
            return False


    def step(self, element, action, counter):
        # done = False
        reward = 0
        actions = ActionChains(self.driver)
        if action == 'click':
            # actions.move_by_offset(coords.get('x') + 1, coords.get('y') + 1).click()
            element.click()
            time.sleep(3)
        if action == 'type':
            # actions.move_by_offset(coords.get('x') + 1, coords.get('y') + 1).send_keys('admin')
            element.send_keys("admin")
            time.sleep(3)
        state_ = self.get_screen(counter)
        counter += 1
        print('reward: ', reward)
        return counter


        # if action == 'click':
        #     actions.move_by_offset(coords[0], coords[1]).click()
        #     state_ = pyautogui.screenshot()
        #     if (self.isTerminal(state_)):
        #         reward += 1
        #         done = True
        #         return state_, reward, done, 'the state is terminal'
        #     else if(state_.isIdenticalStart(state_)):
        #         reward -= 1
        #         return state_, reward, done, nothing changed
        # if action == 'type':
        #     actions.move_by_offset(coords[0], coords[1]).send_keys(keys)
        #     state_ = pyautogui.screenshot()
        #     if (not self.isTerminal(state_) and self.haveCredentials(state_)):
        #         reward += 1
        #         return state_, reward, done, 'typed credentials'
        #     else if (self.isIdenticalStart(state_)):
        #         reward -= 1
        #         return state_, reward, done, 'nothing changed'





