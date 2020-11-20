import time

import cv2
# from PIL import Image
from selenium import webdriver
import numpy as np
import pytesseract

from actor_critic_outer_env.agent import Agent
from actor_critic_outer_env.environment import WebEnv


# env = WebEnv()
# action_space = env.action_space
# print(len(action_space))
agent = Agent(alpha=1e-5, n_actions=2)
observation = cv2.imread(fr'D:\gui-test\resources\initial.png')
action = agent.choose_action(observation)



# selenium_driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# selenium_driver.get('https://digital.sberbank.kz/customer/login')


# Need to get coords element from action

# GET SCREENSHOT FROM ENV
# env = WebEnv()

# time.sleep(5)
# screen, counter = env.get_screen(0)

# login = env.driver.find_element_by_id('log-name')
# password = env.driver.find_element_by_id('log-pass')
#
# login_location = login.location
# login_size = login.size
#
# password_location = password.location
# password_size = password.size
#
# print('login_location: ', login_location)
# print('login_size: ', login_size)
#
# print('password_location: ', password_location)
# print('password_size: ', password_size)
#
# button = env.driver.find_element_by_id('auth-login')
# print('button location: ', button.location)
# print('button coords X: ', button.location.get('x'))
# print('button coords Y: ', button.location.get('y'))
# # TESTING STEP FUNCTION
# # type login
# counter = 0
# reward = 0
# counter = env.step(login, 'type', counter)
#
# # type password
# counter = env.step(password, 'type', counter)
# #
# # # click
# counter = env.step(button, 'click', counter)
# print(env.is_terminal())

# CHECK WILL THE SCREEN BY SELENIUM BE THE SAME?
# state_, reward, done = env.step()



# READ TEXT IN IMAGES WITH CV2 AND PYTESSERACT
# pytesseract.pytesseract.tesseract_cmd = fr'C:\Users\rengu\.conda\envs\guit-test-python\tesseract.exe'
# terminal = cv2.imread(fr'D:\gui-test\resources\terminal_state.png')
# screen = cv2.imread(fr'D:\gui-test\resources\learning_screens\0.png')
# terminal2 = cv2.imread(fr'D:\gui-test\resources\terminal_state_grey.png')
# text = pytesseract.image_to_string(terminal)
# text2 = pytesseract.image_to_string(terminal2)
# text = text.encode('ascii')
# text2 = text2.encode('ascii')
# print(text)
# print(text2)

# CHECK IF TWO IMAGES ARE EQUAL
# difference = cv2.subtract(terminal, screen)
# result = not np.any(difference)
# if result is True:
#     print("Pictures are the same")
# else:
#     cv2.imwrite(fr'resources\ed.jpg', difference )
#     print("Pictures are different, the difference is stored as ed.jpg")


# TEST SCREENSHOT SAVING
# env = WebEnv()
# counter = 0
# while counter < 2:
#     screen, counter = env.get_screen(counter)
#     print('screen size: ', screen)

# element = selenium_driver.find_element_by_class_name('mainPageBlock')
# counter = 0
# filename = "%d" % counter
# state = element.screenshot(fr'resources\learning_screens\{filename}.png')

# initial_state = Image.open(fr'D:\gui-test\resources\0.png')
# resize_state = initial_state.resize((256, 256))
# greyscale_state = resize_state.convert('L')
# greyscale_state.save(fr'resources\terminal_state_grey.png')
# image = Image.open(fr'D:\gui-test\resources\learning_screens\{filename}.png')
# new_image = image.resize((400, 400))
# greyscale_image = new_image.convert('L')
# greyscale_image.save(fr'resources\learning_screens\{filename}.png')


# print(image.size) # Output: (1200, 776)
# print(new_image.size) # Output: (400, 400)