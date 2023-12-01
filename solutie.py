import cv2 as cv
import os
import numpy as np
import imutils


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


configuratie_gasita = []


def determina_configuratie_careu_olitere(img_hsv, lines_horizontal, lines_vertical, img_original):
    matrix = np.empty((15, 15), dtype='str')
    show_image("mask_hsv", img_hsv)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            patch_original = img_original[x_min:x_max, y_min:y_max].copy()

            show_image("mask_hsv", patch_original)
            files = os.listdir('imagini_auxiliare/horizontal')
            for file in files:
                if file[-3:] != 'png': continue;
                template_path = 'imagini_auxiliare/horizontal/' + file
                print(template_path)
                gasit = False
                template = cv.imread(template_path)
                # rotit de 3 ori la 90 de grade
                for i in range(0, 3):
                    template_rotated = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)
                    template_rotated = cv.cvtColor(template_rotated, cv.COLOR_BGR2GRAY)

            # Medie_patch = np.mean(patch)
            # if Medie_patch > 100:
            #     show_image("mask_hsv", patch)
            #     matrix[i][j] = "1"  # to do
            # else:
            #     matrix[i][j] = "o"
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
for lines in range(0, 1500, 100):
    for i in range(0, 1500, 100):
        l = []
        l.append((lines, i))
        l.append((1500, i))
        lines_horizontal.append(l)

lines_vertical = []
for lines in range(0,1500, 100):
    for i in range(0, 1500, 200):
        l = []
        l.append((i, lines))
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

            # gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            # edges = cv.Canny(gray, 50, 150, apertureSize=3)

            # blur = cv.GaussianBlur(edges, (5, 5), 0)

            # show_image('blur', blur)

            print(lines_horizontal)
            print(lines_vertical)

            for lines in range(0, 1500, 100):
                for columns in range(0, 1400, 200):
                    patch = result[lines:lines+100, columns:columns+200].copy()

                    full_imag = result[0:1500, 0:1500].copy()
                    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    edges = cv.Canny(gray, 50, 150, apertureSize=3)
                    blur = cv.GaussianBlur(edges, (5, 5), 0)
                    mean = np.mean(edges)
                    if mean <= 40:
                        show_image('patch', patch)

                    print(mean)

                    full_gray = cv.cvtColor(full_imag, cv.COLOR_BGR2GRAY)
                    full_edges = cv.Canny(full_gray, 10, 50, apertureSize=3)
                    full_blur = cv.GaussianBlur(full_edges, (5, 5), 0)
                    #show_image('full', full_blur)

            # for i in range(0, 1500-1, 100):
            #     for j in range(0, 1500-1, 100):
            #         patch = result[i:x_max, y_min:y_max].copy()
            #         print(y_min, y_max, x_max, x_max)
            #
            #         img_cu_cercuri = detecteaza_cercuri(patch)
            #         print(img_cu_cercuri)
            #         break
                    # if detecteaza_cercuri(patch) is not None:
                    #     print(detecteaza_cercuri(patch))
                    #     detecteaza_linii(detecteaza_cercuri(patch))

            nr_mutare = nr_mutare + 1

            # low_yellow = (0, 0, 232)
            # high_yellow = (123, 255, 255)
            # img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
            # mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
            # matrice = determina_configuratie_careu_olitere(mask_yellow_hsv, lines_horizontal, lines_vertical, result)
            # nr_mutare = nr_mutare + 1


def detecteaza_linii(img):
    print(img)
    show_image('img', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)

    show_image('lines_edges', lines_edges)


def detecteaza_cercuri(img):
    exist_circles = False

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    show_image('blurred', blurred)

    # Use the Hough Circle Transform to detect circles
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,  # Inverse ratio of the accumulator resolution to the image resolution (1 means the same resolution)
        minDist=1,  # Minimum distance between the centers of detected circles
        param1=600,  # Higher threshold for the internal Canny edge detector
        param2=27,  # Threshold for circle detection (lower means more circles will be detected)
        minRadius=1,  # Minimum radius of the detected circles
        maxRadius=20  # Maximum radius of the detected circles
    )

    # If circles are found, draw them on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    print("circles=", circles)
    if circles is not None:
        return img
    else:
        return None
    # Display the result
    # cv.imshow('Detected Circles', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # files = os.listdir('imagini_auxiliare/horizontal')
    # for file in files:
    #     if file[-3:] != 'png': continue;
    #     template_path = 'imagini_auxiliare/horizontal/' + file
    #     print(template_path)
    #     template = cv.imread(template_path)
    #     template_rotated = cv.rotate(template, cv.ROTATE_180)
    #     template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    #     template_rotated = cv.cvtColor(template_rotated, cv.COLOR_BGR2GRAY)
    #
    #

    #     loc = False
    #     threshold = 0.8
    #     w, h = template.shape[::-1]
    #     for scale in np.linspace(0.1, 1.0, 20)[::-1]:
    #         resized = imutils.resize(template, width=int(template.shape[1] * scale))
    #         w, h = resized.shape[::-1]
    #         res = cv.matchTemplate(img_gray, resized, cv.TM_CCOEFF_NORMED)
    #
    #         loc = np.where(res >= threshold)
    #         if len(list(zip(*loc[::-1]))) > 0:
    #             break
    #
    #     if loc and len(list(zip(*loc[::-1]))) > 0:
    #         for pt in zip(*loc[::-1]):
    #             cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    #
    # show_image('result', img)

    # Export the result
    # cv2.imwrite('result_image.jpg', result_image)
    # show_image('template', template)
    # show_image('template_rotated', template_rotated)
    # w, h = template.shape[::-1]
    # res = cv.matchTemplate(img_gray, template, cv.TM_CCORR_NORMED)
    # res_rotated = cv.matchTemplate(img_gray, template_rotated, cv.TM_CCORR_NORMED)
    # show_image('template', template)
    # show_image('template_rotated', template_rotated)
    # threshold = 0.5
    # if template_path == 'imagini_auxiliare/horizontal/4_0.jpg':
    #     print(res_rotated)
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    #
    # loc_rotated = np.where(res_rotated >= threshold)
    # for pt in zip(*loc_rotated[::-1]):
    #     cv.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # cv.imwrite('res.jpg', result)
    # img_res = cv.imread('res.jpg')
    # show_image('res', img_res)
    # lines_horizontal = []
    # for i in range(0, 1500, 200):
    #     l = []
    #     l.append((0, i))
    #     l.append((1500, i))
    #     lines_horizontal.append(l)
    #
    # lines_vertical = []
    # for i in range(0, 1500, 100):
    #     l = []
    #     l.append((i, 0))
    #     l.append((i, 1500))
    #     lines_vertical.append(l)
    #
    # for i in range(len(lines_horizontal) - 1):
    #     for j in range(len(lines_vertical) - 1):
    #         y_min = lines_vertical[j][0][0] + 5
    #         y_max = lines_vertical[j + 1][1][0] - 5
    #         x_min = lines_horizontal[i][0][1] + 5
    #         x_max = lines_horizontal[i + 1][1][1] - 5
    #         patch = result[x_min:x_max, y_min:y_max].copy()
    #
    #         img_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    #
    #         show_image('img_gray', img_gray)
    #
    #         files = os.listdir('imagini_auxiliare/horizontal')
    #         for file in files:
    #             template_path = 'imagini_auxiliare/horizontal/' + file
    #             print(template_path)
    #             template = cv.imread(template_path)
    #             template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    #             res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    #             threshold = 0.8
    #             loc = np.where(res >= threshold)
    #             if len(loc[0]) > 0:
    #                 print("Template found in the image!")
    #                 show_image('template match', patch)


# extrage_piese_imag_aux()

show_images(careu=1)
