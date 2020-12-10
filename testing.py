import time
import cv2
import matplotlib
import numpy as np
from PIL import Image
import tensorflow as tf
from result.environment import Environment

# ******** CHECK IF STATE IS TERMINAL ********
env = Environment()
time.sleep(7)
counter = 0
counter, im_gray, thresh = env.get_screen(counter)
print(type(im_gray))
PIL_image = Image.fromarray(im_gray)
PIL_image.save(fr"D:\gui-test\result\resources\0.png")
# cv2.imwrite(fr'\result\0.png', im_gray)

if env.is_terminal(PIL_image):
    print('The state is terminal')
else:
    print("No it isn't terminal")

# ******** CHECK IF AFTER ACTION NOTHING CHANGES********
# env = Environment()
# time.sleep(7)
# counter = 0
# counter, im_gray, thresh = env.get_screen(counter)
# cv2.imwrite(fr'/result/resources/initial.png', im_gray)
# initial = Image.open(fr'D:\gui-test\result\resources\initial.png')
# initial = tf.keras.preprocessing.image.img_to_array(initial, data_format=None, dtype=None)
# arr1 = np.array(initial)
# state = tf.keras.preprocessing.image.img_to_array(im_gray, data_format=None, dtype=None)
# arr2 = np.asarray(state)
# print(np.array_equal(arr1, arr2))

# ******** GET GUI ELEMENTS COUNTUR ********
# image_path = fr"D:\gui-test\test.png"
# im = cv2.imread(image_path)
# im_copy = im.copy()
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(im_gray, 235, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow(fr'D:\gui-test\contours.png', thresh)
# cv2.waitKey(0)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# for i in range(len(contours)):
#     cnt = contours[i]
#     cv2.drawContours(im_copy, [cnt], 0, (0, 0, 255), 2)

# cnt = contours[3]
# cv2.drawContours(im_copy, [cnt], 0, (255, 0, 0), 2)
# cv2.imshow('Contours', im_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ******** GET GUI ELEMENTS CENTROID ********
# M = cv2.moments(thresh)
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])

# cv2.circle(im_copy, (cX, cY), 5, (255, 0, 0), 2)
# cv2.putText(im_copy, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# centroids = []
# for i in range(len(contours)):
#     Cx = 0
#     Cy = 0
#     for j in range(len(contours[i])):
#         Cx += contours[i][j][0][0]
#         Cy += contours[i][j][0][1]
#
#     Cx = int(Cx / len(contours[i]))
#     Cy = int(Cy / len(contours[i]))
#
#     C = [Cx, Cy]
#     centroids.append(C)
#
# print(centroids)
#
# for i in range(len(centroids)):
#     cX = centroids[i][0]
#     cY = centroids[i][1]
#     cv2.circle(im_copy, (cX, cY), 5, (255, 0, 0), 1)

# cv2.circle(im_copy, (221, 218), 5, (255, 0, 0), 1)
# cv2.circle(im_copy, (201, 192), 5, (0, 255, 0), 1)
# cv2.circle(im_copy, (151, 192), 5, (0, 0, 255), 1)
# cv2.circle(im_copy, (52, 194), 5, (255, 0, 0), 1)
# cv2.circle(im_copy, (111, 105), 5, (0, 255, 0), 1)
# cv2.circle(im_copy, (121, 26), 5, (0, 0, 255), 1)

# cv2.imshow("Image", im_copy)
# cv2.waitKey(0)
# ******** GET GUI CONTOURS WITH MASK ********
# image = cv2.imread(image_path)
# imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#
# (thresh, blackAndWhiteImage) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('Black white image', blackAndWhiteImage)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# print('thresh', thresh)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# lower = np.array([0, 0, 0])
# upper = np.array([255, 255, 255])
# shapeMask = cv2.inRange(image, lower, upper)
# print(shapeMask[0])

# ******** FLOOD FILL ********
# flood = thresh1
# seed = (150, 50)
# cv2.floodFill(flood, None, seedPoint=seed, newVal=(36, 25, 12), loDiff=(0, 255, 0, 0), upDiff=(0, 255, 0, 0))
# cv2.circle(flood, seed, 2, (36, 25, 12), cv2.FILLED, cv2.LINE_AA);
# cv2.imshow('flood', flood)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ******** MAKE SCREENSHOT ********
# import pyautogui
# from result.environment import Environment
# from PIL import Image
# import PIL
#
# env = Environment()
# time.sleep(10)
# image = pyautogui.screenshot(region=(320, 545, 445, 265))
# im = image.resize((224, 224))
# im.save(fr"D:\gui-test\test.png")


# counter = 0
# counter, state = env.get_screen(counter)
#
# x = []
# y = []
# n = 256 * 256
#
# for i in range(256):
#     x.append(i)
#     y.append(i)
#
# c = [[x0, y0] for x0 in x for y0 in y]
#
# i = 0
# score = 0
# counter = 0
# while i < 256:
#     action = random.choice(env.action_space)
#     if action == 'type' and counter >= 1:
#         j = 256 - i
#         while j < 256:
#             c[j][1] = c[j][1] + 50
#             j += 1
#     state_, reward, done, counter = env.step(action, c[i], counter)
#     score += reward
#     if done == True:
#         break
#     print('episode', i, 'action: ', action, 'score: ', score)
#     i += 1


# image = Image.open(fr"result\resources\learning_screens\initial.png")
# strs = pytesseract.image_to_string(image)
# strs = unidecode(strs)
# str = "Hepephi nome vit naponis"
# if strs.__contains__(str):
#     print(True)


# pyautogui.moveTo(350, 555)
# pyautogui.click()
# pyautogui.write('admin')
# time.sleep(2)
# pyautogui.moveTo(350, 670)
# pyautogui.click()
# pyautogui.write('admin')
# time.sleep(2)
# pyautogui.moveTo(350, 770)
# pyautogui.click()
# time.sleep(2)

# state = tf.keras.preprocessing.image.img_to_array(observation, data_format=None, dtype=None)
# print(state)


# terminal = Image.open(fr'D:\gui-test\resources\terminal.png')
# terminal = tf.keras.preprocessing.image.img_to_array(terminal, data_format=None, dtype=None)
# terminal = np.array(terminal)
# print(terminal)

# array = tf.keras.preprocessing.image.img_to_array(state)

# from keras.applications.resnet50 import ResNet50
#
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
# x = base_model.output
# print('x', x)
# x = GlobalAveragePooling2D()(x)
# x = Dense(10, activation='softmax')(x)
# model = Model(base_model.input, x)
# print('model', model)


# state = np.array(state)
# print('state', state)
# normalized_metrics = normalize(state, axis=0, norm='l1')
# print('norm', normalized_metrics)
# for i in range(state[0]):
#     for j in range(state[0]):
#         state[i][j] = float(state[i][j])/float(255)


# print(state)
# arr = np.array(state)
# print('array', arr)
# shape = arr.shape
# print('shape', shape)
# flat_arr = arr.ravel()
# print('flat_arr', flat_arr)
# result_arr = []
# for i in range(len(flat_arr)):
#     print('arr[i]', flat_arr[i], float(flat_arr[i])/float(255))
#     result_arr.append(float(flat_arr[i])/float(255))
#
# for i in range(len(result_arr)):
#     print('result_arr', result_arr[i])


# print(env.isTerminal(state))
# print(np.array(state))
# tensor = tf.convert_to_tensor(state)
# print(tensor)
# env.reset()
#
#
#
#

# print(env.equal(terminal, initial))

# if(env.isTerminal(env, screen)):
#     print("Done")

# ********** TEST STEP **********

# observation_, reward, done, info = env.step(action)
# pyautogui.moveTo(132, 323)
# pyautogui.write('admin')
# pyautogui.moveTo(132, 394)
# pyautogui.click()

# ********** TEST CREATE ARRAY OF COORDINATES  **********


# env = WebEnv()
# env.step([133, 405], 'click', 2)
# options = Options()
# options.add_argument(fr"--load-extension=C:\Users\rengu\AppData\Local\Google\Chrome\User "
#                      fr"Data\Default\Extensions\bihmplhobchoageeokmgbdihknkjbknd\4.1.0_0\manifest.json")


# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--proxy-server=1.2.3.4:8080')

# selenium_driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# selenium_driver.get('https://digital.sberbank.kz/customer/login')
# time.sleep(5)
# actions = ActionChains(selenium_driver)
#
# actions.move_by_offset(132 + 2, 262 + 2).send_keys("admin").perform()
# time.sleep(2)
# actions.move_by_offset(132+2, 343+2).send_keys("admin").perform()
# time.sleep(2)
# actions.move_by_offset(132 + 3, 404 + 3).click().perform()


# # ********** TEST GET COORDINATES FROM IMAGE **********
# observation = cv2.imread(fr'D:\gui-test\resources\initial.png')
# action_space = ['click', 'type']
# gray = cv2.cvtColor(cv2.UMat(observation), cv2.COLOR_BGR2GRAY)
# yx_coords = np.column_stack(np.where(gray >= 0))
# print(yx_coords)
# print(len(yx_coords))
# print(yx_coords[1])
# print(yx_coords[1][0])


# ********** TEST GET RANDOM ACTION **********
# action_space = ['click', 'type']
# for i in range(10):
#     action = random.choice(action_space)
#     print(action)

# ********** GET CHROME PAGE **********
# selenium_driver = webdriver.Chrome(fr'D:\chromedriver.exe')
# selenium_driver.get('https://digital.sberbank.kz/customer/login')

# ********** GET SCREENSHOT FROM ENV **********
# env = WebEnv()
# time.sleep(5)
# screen, counter = env.get_screen(0)
# login = env.driver.find_element_by_id('log-name')
# password = env.driver.find_element_by_id('log-pass')
# login_location = login.location
# login_size = login.size
# password_location = password.location
# password_size = password.size
# print('login_location: ', login_location)
# print('login_size: ', login_size)
# print('password_location: ', password_location)
# print('password_size: ', password_size)
# button = env.driver.find_element_by_id('auth-login')
# print('button location: ', button.location)
# print('button coords X: ', button.location.get('x'))
# print('button coords Y: ', button.location.get('y'))
# button.click()

# ********** TESTING STEP FUNCTION **********
# counter = 0
# reward = 0
# counter = env.step(login, 'type', counter)
# counter = env.step(password, 'type', counter)
# counter = env.step(button, 'click', counter)
# print(env.is_terminal())


# ********** READ TEXT IN IMAGES WITH CV2 AND PYTESSERACT **********
# pytesseract.pytesseract.tesseract_cmd = fr'C:\Users\rengu\.conda\envs\guit-test-python\tesseract.exe'
# terminal = cv2.imread(fr'D:\gui-test\resources\terminal_state.png')
# screen = cv2.imread(fr'D:\gui-test\resources\learning_screens\terminal.png')
# terminal2 = cv2.imread(fr'D:\gui-test\resources\terminal_state_grey.png')
# text = pytesseract.image_to_string(terminal)
# text2 = pytesseract.image_to_string(terminal2)
# text = text.encode('ascii')
# text2 = text2.encode('ascii')
# print(text)
# print(text2)

# ********** CHECK IF TWO IMAGES ARE EQUAL **********
# difference = cv2.subtract(terminal, screen)
# result = not np.any(difference)
# if result is True:
#     print("Pictures are the same")
# else:
#     cv2.imwrite(fr'resources\ed.jpg', difference )
#     print("Pictures are different, the difference is stored as ed.jpg")


# ********** TEST SCREENSHOT SAVING **********
# env = WebEnv()
# counter = 0
# while counter < 2:
#     screen, counter = env.get_screen(counter)
#     print('screen size: ', screen)
# element = selenium_driver.find_element_by_class_name('mainPageBlock')
# counter = 0
# filename = "%d" % counter
# state = element.screenshot(fr'resources\learning_screens\{filename}.png')
# initial_state = Image.open(fr'D:\gui-test\resources\terminal.png')
# resize_state = initial_state.resize((256, 256))
# greyscale_state = resize_state.convert('L')
# greyscale_state.save(fr'resources\terminal_state_grey.png')
# image = Image.open(fr'D:\gui-test\resources\learning_screens\{filename}.png')
# new_image = image.resize((400, 400))
# greyscale_image = new_image.convert('L')
# greyscale_image.save(fr'resources\learning_screens\{filename}.png')
# print(image.size) # Output: (1200, 776)
# print(new_image.size) # Output: (400, 400)
