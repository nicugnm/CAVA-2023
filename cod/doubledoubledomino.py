import cv2 as cv
import numpy as np
import os

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extrage_careu(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    low_yellow = (0, 0, 0)
    high_yellow = (255, 140, 255)

    img_hsv = cv.cvtColor(image.copy(), cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    contours, _ = cv.findContours(mask_yellow_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 810
    height = 810

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    show_image("detected corners",image_copy)
    return
    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

    return result

lines_horizontal=[]
for i in range(0,811,90):
    l=[]
    l.append((0,i))
    l.append((809,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,811,90):
    l=[]
    l.append((i,0))
    l.append((i,809))
    lines_vertical.append(l)

files=os.listdir('data/train')
for file in files:
    if file[-3:]=='jpg':
        img = cv.imread('data/train/'+file)
        result=extrage_careu(img)
        for line in  lines_vertical :
            cv.line(result, line[0], line[1], (0, 255, 0), 5)
        for line in  lines_horizontal :
            cv.line(result, line[0], line[1], (0, 0, 255), 5)
        show_image('img',result)