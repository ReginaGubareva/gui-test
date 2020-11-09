# Here should be methods for selenium

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


class SeleniumAction:

    def action(self, coordinates, action, keys=''):
        driver = webdriver.Firefox(firefox_binary='/usr/bin/firefox')
        driver.get("https://www.google.com/?hl=en")
        actions = ActionChains(driver)
        Y_coordinates = 382 #coordinates[0]
        X_coordinates = 396 #coordinates[0]
        actions.move_to_element_with_offset(driver.find_element_by_tag_name('body'), 0, 0)
        if action == 'CLICK':
            actions.move_by_offset(X_coordinates, Y_coordinates).click()
        if action == 'TYPE':
            actions.move_by_offset(X_coordinates, Y_coordinates).send_keys(keys)
