import gym
from PIL import Image
from selenium import webdriver
from gym import spaces
import pyautogui
import webbrowser
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageChops

class Environment:
    def __init__(self):
        self.chrome = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
        self.url = fr'https://digital.sberbank.kz/customer/login'
        webbrowser.get(self.chrome).open(self.url)
        self.action_space = ['click', 'type']

    def step(self, action, coordinates, counter):
        done = False
        reward = 0
        counter, state = self.get_screen(counter)
        if action == "click":
            pyautogui.moveTo(coordinates[0] + 325, coordinates[1] + 545)
            pyautogui.click()
            counter, state = self.get_screen(counter)
            reward = 1
            if self.isTerminal(state):
                done = True
                reward = 1
                print("Done")
        if action == "type":
            pyautogui.moveTo(coordinates[0] + 325, coordinates[1] + 545)
            pyautogui.write("admin")
            counter, state = self.get_screen(counter)
            reward = 1
            if self.isTerminal(state):
                reward = 1
                done = True
                print("Done")
        return state, reward, done, counter


    def reset(self):
        pyautogui.hotkey('f5')



    def get_screen(self, counter):
        img = pyautogui.screenshot(region=(320, 545, 445, 265))
        observation = img.convert('L').resize((256, 256))
        filename = "%d" % counter
        observation.save(fr'resources\learning_screens\{filename}.png')
        counter += 1
        return counter, observation

    @staticmethod
    def isTerminal(state):
        terminal = Image.open(fr'D:\gui-test\resources\terminal.png')
        arr1 = np.array(terminal)
        arr2 = np.array(state)
        return np.array_equal(arr1, arr2)


