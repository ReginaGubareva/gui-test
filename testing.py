import random
import time
import tensorflow as tf
import cv2
# from PIL import Image
from selenium import webdriver
import numpy as np
import pytesseract
import imutils
from actor_critic_outer_env.agent import Agent
from actor_critic_outer_env.environment import WebEnv
from itertools import product
import pandas as pd

rows_count = 256
cols_count = 2
action_space = ['click', 'type']
coords = [[0 for j in range(cols_count)] for i in range(rows_count)]
for i in range(rows_count):
    for j in range(256):
        coords[i][0] = i
        coords[i][1] = j

print(coords)


# for i in range(len(yx_coords)):
#     action = random.choice(action_space)
#     print('coords: ', yx_coords[i], 'action: ', action)
# random_action = random.choice(action_space)
# env = WebEnv()
# env.step(yx_coords[1], random_action, 0)

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
# screen = cv2.imread(fr'D:\gui-test\resources\learning_screens\0.png')
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
