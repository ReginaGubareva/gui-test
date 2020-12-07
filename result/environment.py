import webbrowser
from unidecode import unidecode
import pyautogui
import pytesseract
import cv2
import time
from PIL import Image
import tensorflow as tf
import numpy as np


class Environment:
    def __init__(self):
        self.chrome = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s"
        self.url = fr'https://digital.sberbank.kz/customer/login'
        webbrowser.get(self.chrome).open(self.url)
        time.sleep(5)
        self.action_space = ['click', 'type']

    def step(self, action, coordinates, counter):
        print('action:', action, '-- coordinates:', coordinates)
        done = False
        reward = 0
        counter, state, th = self.get_screen(counter)
        if action == "click":
            pyautogui.moveTo(coordinates[0] + 327, coordinates[1] + 550)
            pyautogui.click()
            counter, state, th = self.get_screen(counter)
            reward = 1
            if self.is_terminal(state):
                done = True
                reward = 3
                print("Done")
        if action == "type":
            pyautogui.moveTo(coordinates[0] + 330, coordinates[1] + 548)
            pyautogui.click()
            pyautogui.write("admin")
            counter, state, th = self.get_screen(counter)
            reward = 1
            if self.is_terminal(state):
                reward = 3
                done = True
                print("Done")
            if self.no_changes(state):
                reward = -1
        return state, reward, done, counter

    def reset(self):
        pyautogui.hotkey('f5')

    def get_screen(self, counter):
        img = pyautogui.screenshot(region=(320, 545, 445, 265))
        observation = img.resize((224, 224))
        filename = "%d" % counter
        observation.save(fr'D:\gui-test\result\resources\learning_screens\{filename}.png')
        image_path = fr"D:\gui-test\result\resources\learning_screens\{filename}.png"
        im = cv2.imread(image_path)
        # observation = im.copy()
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(im_gray, 235, 255, cv2.THRESH_BINARY_INV)
        counter += 1
        return counter, im_gray, thresh

    def get_contours(self, observation, thresh):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of Contours found = " + str(len(contours)))
        centroids = []
        for i in range(len(contours)):
            Cx = 0
            Cy = 0
            for j in range(len(contours[i])):
                Cx += contours[i][j][0][0]
                Cy += contours[i][j][0][1]

            Cx = int(Cx / len(contours[i]))
            Cy = int(Cy / len(contours[i]))

            C = [Cx, Cy]
            centroids.append(C)

        print(centroids)
        return contours, centroids

    @staticmethod
    def is_terminal(state):
        strs = pytesseract.image_to_string(state)
        strs = unidecode(strs)
        str = "Hepephi nome vit naponis"
        if strs.__contains__(str):
            print('the terminal state')
            return True
        else:
            return False

    @staticmethod
    def no_changes(state):
        initial = Image.open(fr'D:\gui-test\result\resources\initial.png')
        initial = tf.keras.preprocessing.image.img_to_array(initial, data_format=None, dtype=None)
        arr1 = np.array(initial)
        state = tf.keras.preprocessing.image.img_to_array(state, data_format=None, dtype=None)
        arr2 = np.asarray(state)
        return np.array_equal(arr1, arr2)

    # **** Terminal method compare two images ****
    # @staticmethod
    # def isTerminal(state):
    #     terminal = Image.open(fr'D:\gui-test\result\resources\terminal.png')
    #     terminal = tf.keras.preprocessing.image.img_to_array(terminal, data_format=None, dtype=None)
    #     arr1 = np.array(terminal)
    #     state = tf.keras.preprocessing.image.img_to_array(state, data_format=None, dtype=None)
    #     arr2 = np.asarray(state)
    #     return np.array_equal(arr1, arr2)


