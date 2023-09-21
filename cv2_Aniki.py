import cv2
import numpy as np


def viewImage(image):
    cv2.namedWindow('donut', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def left_conter(image):
    viewImage(image)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([0, 45, 125])
    yellow_high = np.array([60, 150, 253])
    curr_mask = cv2.inRange(hsv_img, yellow_low, yellow_high)
    hsv_img[curr_mask > 0] = ([0, 0, 0])
    viewImage(hsv_img)
    mass = []
    count = 0
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray, 50, 255, 0)
    threshold = 255 - threshold
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        mass.append([len(contours[count]), count])
        count += 1
    mass.sort(key=lambda x: x[0], reverse=True)
    cv2.drawContours(image, contours[mass[5][1]], -1, (0, 0, 255), 3)
    x = np.sum(contours[mass[5][1]][:, :, 0]) / len(contours[mass[5][1]])
    print(x)
    y = np.sum(contours[mass[5][1]][:, :, 1]) / len(contours[mass[5][1]])
    print(y)
    color_yellow = (0, 255, 255)
    cv2.putText(image, "2", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
    viewImage(image)
    return contours[mass[5][1]]


def right_conter(image):
    viewImage(image)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([10, 45, 125])
    yellow_high = np.array([60, 100, 253])
    curr_mask = cv2.inRange(hsv_img, yellow_low, yellow_high)
    hsv_img[curr_mask > 0] = ([0, 0, 0])
    viewImage(hsv_img)
    mass = []
    count = 0
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray, 50, 255, 0)
    threshold = 255 - threshold
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        mass.append([len(contours[count]), count])
        count += 1
    mass.sort(key=lambda x: x[0], reverse=True)
    cv2.drawContours(image, contours[mass[2][1]], -1, (0, 0, 255), 3)
    x = np.sum(contours[mass[2][1]][:, :, 0]) / len(contours[mass[2][1]])
    print(x)
    y = np.sum(contours[mass[2][1]][:, :, 1]) / len(contours[mass[2][1]])
    print(y)
    color_yellow = (0, 255, 255)
    cv2.putText(image, "3", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
    viewImage(image)
    return contours[mass[2][1]]


image = cv2.imread('girls22.png')
viewImage(image)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img)

blue_low = np.array([12, 105, 100])
blue_high = np.array([90, 255, 255])
curr_mask = cv2.inRange(hsv_img, blue_low, blue_high)
hsv_img[curr_mask > 0] = ([120, 255, 255])
viewImage(hsv_img)

RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
viewImage(gray)

ret, threshold = cv2.threshold(gray, 110, 255, 0)
viewImage(threshold)

threshold = 255 - threshold
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
x = np.sum(contours[-1][:, :, 0]) / len(contours[-1])
y = np.sum(contours[-1][:, :, 1]) / len(contours[-1])
color_yellow = (0, 255, 255)
cv2.putText(image, "1", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)
cv2.drawContours(image, contours[-1], -1, (0, 0, 255), 3)
viewImage(image)
contours_2 = left_conter(image)
contours_3 = right_conter(image)
viewImage(image)
cv2.fillPoly(image, pts=[contours[-1]], color=(0, 255, 255))
cv2.fillPoly(image, pts=[contours_2], color=(255, 255, 0))
cv2.fillPoly(image, pts=[contours_3], color=(255, 0, 255))
viewImage(image)
