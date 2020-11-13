# import multiprocessing
# import threading
# import tensorflow as tf
# import numpy as np
# import gym
# import os
# import shutil
# import matplotlib.pyplot as plt
# from selenium import webdriver
#
# OUTPUT_GRAPH = True
# LOG_DIR = r'/log'
# N_WORKERS = multiprocessing.cpu_count()
# MAX_GLOBAL_EP = 1000
# GLOBAL_NET_SCOPE = 'Global_Net'
# UPDATE_GLOBAL_ITER = 10
# GAMMA = 0.9
# ENTROPY_BETA = 0.001
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.001    # learning rate for critic
# GLOBAL_RUNNING_R = []
# GLOBAL_EP = 0
#
# #initialize environment
# driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# env = driver.get('https://digital.sberbank.kz/')
#
# class ACNet():