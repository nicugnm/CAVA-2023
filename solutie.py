import cv2
import cv2 as cv
import os
import numpy as np

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
            if careu == 0:
                show_image('img', img)
                show_image('resized', imagine_decupata(img))
            else:
                img = continut_mijloc_alb(img)
                arata_careu(imagine_decupata(img))

            nr_mutare = nr_mutare + 1

def imagine_decupata(image):
    height, width, _ = image.shape
    return image[1150:height - 1150, 720:width - 600]

def continut_mijloc_alb(image):
    height, width, _ = image.shape

    # Definirea zonei din mijloc pe care dorești să o faci alb
    center_box_width = 1000  # Lățimea zonei
    center_box_height = 1000  # Înălțimea zonei

    # Calculează coordonatele colțului stânga-sus al zonei din mijloc
    start_x = (width - center_box_width) // 2
    start_y = (height - center_box_height) // 2

    # Creează o mască pentru zona din mijloc (grayscale)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[start_y:start_y + center_box_height, start_x:start_x + center_box_width] = 255

    # Setează zona din mijloc la alb în imagine
    image[mask == 255] = [0, 0, 0]

    return image

def extrage_careu(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 0.9, image_g_blur, -0.4, 0)
    show_image('image_sharpened',image_sharpened)
    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel)
    show_image('image_thresholded',thresh)

    # edges = cv.Canny(thresh, 300, 400)
    # show_image('edges',edges)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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

def arata_careu(img):
    lines_horizontal = []
    for i in range(0, 811, 90):
        l = []
        l.append((0, i))
        l.append((809, i))
        lines_horizontal.append(l)

    lines_vertical = []
    for i in range(0, 811, 90):
        l = []
        l.append((i, 0))
        l.append((i, 809))
        lines_vertical.append(l)


    result = extrage_careu(img)
    for line in lines_vertical:
        cv.line(result, line[0], line[1], (0, 255, 0), 5)
    for line in lines_horizontal:
        cv.line(result, line[0], line[1], (0, 0, 255), 5)
    show_image('img', result)

show_images(careu=1)