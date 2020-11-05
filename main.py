import datetime
import time
import pyautogui
from selenium import webdriver
from selenium.webdriver import ActionChains

num_of_episodes = 100
time_step = 2
t = datetime.now()
t_max = 20

# open the site
driver = webdriver.Chrome(fr'D:\chromedriver.exe')
driver.get('https://digital.sberbank.kz/customer/login')
time.sleep(3)  # Let the user actually see something!
for _ in range(num_of_episodes):
    t_start = t
    counter = 1
    while t - t_start == t_max:
        s_t = pyautogui.screenshot()

        # save screenshots
        filename = "screen%d" % counter
        s_t.save(fr"D:\gui-test-python\resources\{filename}.png")
        counter += 1

        b_t = "visual boundaries of the DOM elements in the current view"

        # For getting per pixel probabilities we should modify algorithm A3C
        # especially, action output
        map_of_probabilities = "feed s_t to A3C and get map of probabilities"

        probabilities_per_element = "the elemtn probabilites"
        element = "choose element from probabilities_per_element"
        # make the action for the element

    # close the connection with browser
    driver.quit()


# To click button in selenium we should
# ActionChains(browser).click(element).perform()
