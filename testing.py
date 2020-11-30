# import os
import random
import time
import tensorflow as tf
import cv2
from PIL import Image
# from selenium import webdriver
import numpy as np
# import pytesseract
# import imutils
# from actor_critic_outer_env.agent import Agent
# from actor_critic_outer_env.environment import WebEnv
# from itertools import product
# import pandas as pd
# from selenium.webdriver import ActionChains
# from selenium.webdriver.chrome.options import Options
import webbrowser
import pyautogui
from actor_critic_keras.actor_critic_keras import Agent
from environment_pyautogui import Environment
from PIL import ImageChops


# pyautogui.moveTo(350, 555)
# pyautogui.click()
# pyautogui.write('admin')
# time.sleep(2)
# pyautogui.moveTo(350, 670)
# pyautogui.click()
# pyautogui.write('admin')
# time.sleep(2)
# pyautogui.moveTo(350, 770)
# pyautogui.click()
# time.sleep(2)

# state = tf.keras.preprocessing.image.img_to_array(observation, data_format=None, dtype=None)
# print(state)




env = Environment()
time.sleep(25)
env.reset()


x = []
y = []
n = 256 * 256

for i in range(256):
    x.append(i)
    y.append(i)

c = [[x0, y0] for x0 in x for y0 in y]
#
# i = 0
# score = 0
# counter = 0
# while i < 256:
#     action = random.choice(env.action_space)
#     state_, reward, done, counter = env.step(action, c[i], counter)
#     score += reward
#     if done == True:
#         break
#     print('episode', i, 'action: ', action, 'score: ', score)
#     i += 1

# print(env.equal(terminal, initial))

# if(env.isTerminal(env, screen)):
#     print("Done")

# ********** TEST STEP **********

# observation_, reward, done, info = env.step(action)
# pyautogui.moveTo(132, 323)
# pyautogui.write('admin')
# pyautogui.moveTo(132, 394)
# pyautogui.click()

# ********** TEST CREATE ARRAY OF COORDINATES  **********


# env = WebEnv()
# env.step([133, 405], 'click', 2)
# options = Options()
# options.add_argument(fr"--load-extension=C:\Users\rengu\AppData\Local\Google\Chrome\User "
#                      fr"Data\Default\Extensions\bihmplhobchoageeokmgbdihknkjbknd\4.1.0_0\manifest.json")


# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--proxy-server=1.2.3.4:8080')

# selenium_driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# selenium_driver.get('https://digital.sberbank.kz/customer/login')
# time.sleep(5)
# actions = ActionChains(selenium_driver)
#
# actions.move_by_offset(132 + 2, 262 + 2).send_keys("admin").perform()
# time.sleep(2)
# actions.move_by_offset(132+2, 343+2).send_keys("admin").perform()
# time.sleep(2)
# actions.move_by_offset(132 + 3, 404 + 3).click().perform()


# # ********** TEST GET COORDINATES FROM IMAGE **********
# observation = cv2.imread(fr'D:\gui-test\resources\initial.png')
# action_space = ['click', 'type']
# gray = cv2.cvtColor(cv2.UMat(observation), cv2.COLOR_BGR2GRAY)
# yx_coords = np.column_stack(np.where(gray >= 0))
# print(yx_coords)
# print(len(yx_coords))
# print(yx_coords[1])
# print(yx_coords[1][0])


# ********** TEST GET RANDOM ACTION **********
# action_space = ['click', 'type']
# for i in range(10):
#     action = random.choice(action_space)
#     print(action)

# ********** GET CHROME PAGE **********
# selenium_driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# selenium_driver.get('https://digital.sberbank.kz/customer/login')

# ********** GET SCREENSHOT FROM ENV **********
# env = WebEnv()
# time.sleep(5)
# screen, counter = env.get_screen(0)
# login = env.driver.find_element_by_id('log-name')
# password = env.driver.find_element_by_id('log-pass')
# login_location = login.location
# login_size = login.size
# password_location = password.location
# password_size = password.size
# print('login_location: ', login_location)
# print('login_size: ', login_size)
# print('password_location: ', password_location)
# print('password_size: ', password_size)
# button = env.driver.find_element_by_id('auth-login')
# print('button location: ', button.location)
# print('button coords X: ', button.location.get('x'))
# print('button coords Y: ', button.location.get('y'))
# button.click()

# ********** TESTING STEP FUNCTION **********
# counter = 0
# reward = 0
# counter = env.step(login, 'type', counter)
# counter = env.step(password, 'type', counter)
# counter = env.step(button, 'click', counter)
# print(env.is_terminal())


# ********** READ TEXT IN IMAGES WITH CV2 AND PYTESSERACT **********
# pytesseract.pytesseract.tesseract_cmd = fr'C:\Users\rengu\.conda\envs\guit-test-python\tesseract.exe'
# terminal = cv2.imread(fr'D:\gui-test\resources\terminal_state.png')
# screen = cv2.imread(fr'D:\gui-test\resources\learning_screens\terminal.png')
# terminal2 = cv2.imread(fr'D:\gui-test\resources\terminal_state_grey.png')
# text = pytesseract.image_to_string(terminal)
# text2 = pytesseract.image_to_string(terminal2)
# text = text.encode('ascii')
# text2 = text2.encode('ascii')
# print(text)
# print(text2)

# ********** CHECK IF TWO IMAGES ARE EQUAL **********
# difference = cv2.subtract(terminal, screen)
# result = not np.any(difference)
# if result is True:
#     print("Pictures are the same")
# else:
#     cv2.imwrite(fr'resources\ed.jpg', difference )
#     print("Pictures are different, the difference is stored as ed.jpg")


# ********** TEST SCREENSHOT SAVING **********
# env = WebEnv()
# counter = 0
# while counter < 2:
#     screen, counter = env.get_screen(counter)
#     print('screen size: ', screen)
# element = selenium_driver.find_element_by_class_name('mainPageBlock')
# counter = 0
# filename = "%d" % counter
# state = element.screenshot(fr'resources\learning_screens\{filename}.png')
# initial_state = Image.open(fr'D:\gui-test\resources\terminal.png')
# resize_state = initial_state.resize((256, 256))
# greyscale_state = resize_state.convert('L')
# greyscale_state.save(fr'resources\terminal_state_grey.png')
# image = Image.open(fr'D:\gui-test\resources\learning_screens\{filename}.png')
# new_image = image.resize((400, 400))
# greyscale_image = new_image.convert('L')
# greyscale_image.save(fr'resources\learning_screens\{filename}.png')
# print(image.size) # Output: (1200, 776)
# print(new_image.size) # Output: (400, 400)
