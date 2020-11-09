import torch
from torch.autograd import Variable
import gym

import pyautogui
from selenium import webdriver
import time



# driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# driver.get('https://digital.sberbank.kz/customer/login')
# time.sleep(3)  # Let the user actually see something!
#
# state = pyautogui.screenshot()
#
# print(torch.from_numpy(state).float())


# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()