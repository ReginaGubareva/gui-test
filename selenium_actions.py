
# Here should be methods for selenium
import pyautogui
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

class SeleniumAction:
    def make_action(self, coordinates, action, keys=''):
        reward = 0
        done = False
        driver = webdriver.Firefox(firefox_binary='/usr/bin/firefox')
        driver.get("https://www.google.com/?hl=en")
        actions = ActionChains(driver)
        Y_coordinates = 382 #coordinates[0]
        X_coordinates = 396 #coordinates[0]
        actions.move_to_element_with_offset(driver.find_element_by_tag_name('body'), 0, 0)
        if action == 'CLICK':
            actions.move_by_offset(X_coordinates, Y_coordinates).click()
            s_t = pyautogui.screenshot()
            if(s_t.isTerminal()):
                # reward += 1
                done = True
            else:
                reward -= 1
        if action == 'TYPE':
            actions.move_by_offset(X_coordinates, Y_coordinates).send_keys(keys)
            s_t = pyautogui.screenshot()
            if(not self.isTerminal(s_t) and self.haveCredentials(s_t)):
                reward += 1
            else:
                reward -= 1

        return reward, done

    def isTerminal(self, state):
        terminal_state_1 = Image.open(r'D:\gui-test\resources\terminal_state_1.PNG')
        terminal_state_2 = Image.open(r'D:\gui-test\resources\start_state.png')
        if(open(state, "rb").read() == open(terminal_state_1,"rb").read()
                or open(state, "rb").read() != open(terminal_state_2, "rb").read()):
            return True
        else:
            return False

    def haveCredentials(s_t):

        return True
