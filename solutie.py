import cv2 as cv
import os
import numpy as np


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


piesa_gasita = []


def determina_configuratie_careu_olitere(img_hsv, lines_horizontal, lines_vertical, img_original):
    matrix = np.empty((15, 15), dtype='str')
    show_image("mask_hsv", img_hsv)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 5
            y_max = lines_vertical[j + 1][1][0] - 5
            x_min = lines_horizontal[i][0][1] + 5
            x_max = lines_horizontal[i + 1][1][1] - 5
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            y_min = lines_vertical[j][0][0] + 8
            y_max = lines_vertical[j + 1][1][0] - 8
            x_min = lines_horizontal[i][0][1] + 8
            x_max = lines_horizontal[i + 1][1][1] - 8
            patch_original = img_original[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch = np.mean(patch)
            if Medie_patch > 50:
                show_image("mask_hsv", patch)
                matrix[i][j] = "1"  # to do
            else:
                matrix[i][j] = "o"
    return matrix


def imagine_decupata(image):
    height, width, _ = image.shape
    return image[1150:height - 1150, 720:width - 600]


def extrage_careu(image):
    low_yellow = (80, 70, 0)
    high_yellow = (255, 255, 255)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)

    contours, _ = cv.findContours(mask_yellow_hsv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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

    width = 1500
    height = 1500

    image_copy = image
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    show_image("detected corners", image_copy)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))

    return result


def arata_careu(img):
    lines_horizontal = []
    for i in range(0, 1500, 200):
        l = []
        l.append((0, i))
        l.append((1500, i))
        lines_horizontal.append(l)

    lines_vertical = []
    for i in range(0, 1500, 200):
        l = []
        l.append((i, 0))
        l.append((i, 1500))
        lines_vertical.append(l)
    result = extrage_careu(img)
    for line in lines_vertical:
        cv.line(result, line[0], line[1], (0, 255, 0), 5)
    for line in lines_horizontal:
        cv.line(result, line[0], line[1], (0, 0, 255), 5)
    show_image('img', result)

    return result


lines_horizontal = []
for i in range(0, 1500, 200):
    l = []
    l.append((0, i))
    l.append((1500, i))
    lines_horizontal.append(l)

lines_vertical = []
for i in range(0, 1500, 100):
    l = []
    l.append((i, 0))
    l.append((i, 1500))
    lines_vertical.append(l)


def show_images(careu):
    files = os.listdir('antrenare')
    nr_joc = 1
    nr_mutare = 1
    for file in files:
        if nr_mutare == 21:
            nr_joc = nr_joc + 1
            nr_mutare = 1
        if nr_joc == 6:
            break
        if file[-3:] == 'jpg':
            image_path = 'antrenare/' + str(nr_joc) + '_' + ('0' if nr_mutare <= 9 else '') + str(nr_mutare) + '.jpg'
            print(image_path)
            img = cv.imread(image_path)
            result = extrage_careu(imagine_decupata(img))

            low_yellow = (0, 0, 232)
            high_yellow = (123, 255, 255)
            img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
            mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
            matrice = determina_configuratie_careu_olitere(mask_yellow_hsv, lines_horizontal, lines_vertical, result)
            nr_mutare = nr_mutare + 1


show_images(careu=1)
