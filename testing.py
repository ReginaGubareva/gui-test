import torch
import pyautogui
# from torch.autograd import Variable
# import gym
# import cv2
# from selenium import webdriver
# import time
# from skimage.transform import resize
# import numpy as np

s_t = pyautogui.screenshot()
state = torch.from_numpy(s_t)
print('from numpy: ', state)
print('float: ', state.float())
print('unsqueeze(0): ', state.float().unsqueeze(0))


