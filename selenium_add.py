import time
import pyautogui
from selenium import webdriver
from selenium.webdriver import ActionChains


driver = webdriver.Chrome(fr'D:\chromedriver.exe')
driver.get('https://digital.sberbank.kz/customer/login')
time.sleep(3)  # Let the user actually see something!
counter = 1
screen_shot = pyautogui.screenshot()
filename = "screen%d" % counter
screen_shot.save(fr"D:\gui-test-python\resources\{filename}.png")
time.sleep(3)
driver.quit()

# To click button in selenium we should
# ActionChains(browser).click(element).perform()

# counter = 1
# while counter > 10:
#     # Open web browser
#     webbrowser.open("https://digital.sberbank.kz/customer/login")
#
#     make_image_black_and_white(pyautogui.screenshot(), counter)
#     # screenShot.save(fr'D:\gui-test-python\resources\{filename}')
#     counter = counter + 1

# num_of_episodes = 100
# time_step = 2
# t = datetime.now()
# t_max = 20
# for _ in range(num_of_episodes):
#     t_start = t
#     while t - t_start == t_max:
#         s_t = "screenshot for the current view"
#         b_t = "visual boundaries of the DOM elements in the current view"
