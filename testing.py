import torch
import pyautogui
# from torch.autograd import Variable
# import gym
# import cv2
from selenium import webdriver
import time
# from skimage.transform import resize
# import numpy as np

driver = webdriver.Chrome(fr'D:\chromedriver.exe')
driver.get('https://digital.sberbank.kz/customer/login')
time.sleep(5)

s_t = pyautogui.screenshot()
s_t.save(fr"D:\gui-test\resources\start_state.png")

# state = torch.from_numpy(s_t)
#
# print('from numpy: ', state)
# print('float: ', state.float())
# print('unsqueeze(0): ', state.float().unsqueeze(0))


