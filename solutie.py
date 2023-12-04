import cv2 as cv
import os
import numpy as np


def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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

    # cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    # cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    # show_image("detected corners", image_copy)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))

    return result


def translate_line_column(line, column):
    line = (line + 100) // 100
    column = (column + 100) // 100
    column = chr(ord('A') + column - 1)

    return line, column


def careu_pct_line_column(line, column):
    line = line - 1
    column_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                      'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14}
    return line, column_mapping[column]


piese_tabla = []
piese_careu_punctaj = []

punctaj_careu = [
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
    [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
    [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
    [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
    [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]
]


def show_images():
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
            nr_joc_template = str(nr_joc) + '_' + ('0' if nr_mutare <= 9 else '') + str(nr_mutare)
            image_path = 'antrenare/' + nr_joc_template + '.jpg'
            img = cv.imread(image_path)
            result = extrage_careu(imagine_decupata(img))

            rezultat_path = 'rezultate/' + nr_joc_template + '.txt'
            fisier = open(rezultat_path, 'w')

            nr_solutii = 0

            for i in range(0, 101, 100):
                # piese pe orizontala
                for lines in range(0, 1500, 100):
                    for columns in range(i, 1500 if i != 0 else 1300, 200):
                        patch = result[lines:lines + 100, columns:columns + 200].copy()
                        are_cercuri = detecteaza_cercuri(patch.copy())[0]

                        # if nr_joc_template == '2_01' or nr_joc_template == '1_20':
                        #    are_cercuri = detecteaza_cercuri(patch.copy(), show=nr_joc_template, lines=lines, columns=columns)[0]

                        if are_cercuri is not None:
                            # show_image('patch', detecteaza_cercuri(patch.copy())[0])
                            img_despartita = patch[0:100, 0:100]
                            img_despartita_2 = patch[0:100, 100:200]

                            are_cercuri_1, nr_cercuri_1 = detecteaza_cercuri(img_despartita)
                            are_cercuri_2, nr_cercuri_2 = detecteaza_cercuri(img_despartita_2)
                            if (are_cercuri_1 is not None) and (are_cercuri_2 is not None):
                                line_1, col_1 = translate_line_column(lines, columns)
                                line_2, col_2 = translate_line_column(lines, columns + 100)

                                if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii < 2):
                                    nr_solutii = nr_solutii + 1
                                    piese_tabla.append((nr_joc, line_1, col_1))
                                    piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                    str_tuplu = str(line_1) + str(col_1) + " " + str(nr_cercuri_1) + str("\n")
                                    fisier.writelines(str_tuplu)
                                    print("Linie si col", line_1, col_1)

                                if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                    nr_solutii = nr_solutii + 1
                                    piese_tabla.append((nr_joc, line_2, col_2))
                                    piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                    str_tuplu2 = str(line_2) + str(col_2) + " " + str(nr_cercuri_2) + str("\n")
                                    fisier.writelines(str_tuplu2)
                                    print("Linie si col", line_2, col_2)

                            if (are_cercuri_1 is None) or (are_cercuri_2 is None):
                                low_yellow_fara_cercuri = (90, 0, 190)
                                high_yellow_fara_cercuri = (255, 100, 255)

                                img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                total_pixels = mask_yellow_hsv_img_despartita_1.size
                                percentage_white = (white_pixels / total_pixels) * 100

                                if (are_cercuri_1 is None) and (percentage_white >= 70):
                                    line_1, col_1 = translate_line_column(lines, columns)

                                    if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii < 2):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_1, col_1))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                        str_tuplu = str(line_1) + str(col_1) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_1, col_1)

                                    img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                    mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                                  low_yellow_fara_cercuri,
                                                                                  high_yellow_fara_cercuri)

                                    white_pixels = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                    total_pixels = mask_yellow_hsv_img_despartita_2.size
                                    percentage_white = (white_pixels / total_pixels) * 100

                                    if (are_cercuri_2 is not None) and (percentage_white >= 60):
                                        line_2, col_2 = translate_line_column(lines, columns + 100)

                                        if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                            nr_solutii = nr_solutii + 1
                                            piese_tabla.append((nr_joc, line_2, col_2))
                                            piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                            str_tuplu2 = str(line_2) + str(col_2) + " " + str(nr_cercuri_2) + str("\n")
                                            fisier.writelines(str_tuplu2)
                                            print("Linie si col", line_2, col_2)

                                img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                total_pixels = mask_yellow_hsv_img_despartita_2.size
                                percentage_white = (white_pixels / total_pixels) * 100

                                if (are_cercuri_2 is None) and (percentage_white >= 70):
                                    line_2, col_2 = translate_line_column(lines, columns + 100)

                                    if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_2, col_2))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                        str_tuplu = str(line_2) + str(col_2) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_2, col_2)

                                    img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                    mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                                  low_yellow_fara_cercuri,
                                                                                  high_yellow_fara_cercuri)

                                    white_pixels = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                    total_pixels = mask_yellow_hsv_img_despartita_1.size
                                    percentage_white = (white_pixels / total_pixels) * 100

                                    if (are_cercuri_1 is not None) and (percentage_white >= 60):
                                        line_1, col_1 = translate_line_column(lines, columns)

                                        if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii < 2):
                                            nr_solutii = nr_solutii + 1
                                            piese_tabla.append((nr_joc, line_1, col_1))
                                            piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                            str_tuplu2 = str(line_1) + str(col_1) + " " + str(nr_cercuri_1) + str("\n")
                                            fisier.writelines(str_tuplu2)
                                            print("Linie si col", line_1, col_1)

                # piese pe verticala
                for lines in range(i, 1500 if i != 0 else 1300, 200):
                    for columns in range(0, 1500, 100):
                        patch = result[lines:lines + 200, columns:columns + 100].copy()

                        are_cercuri = detecteaza_cercuri(patch.copy())[0]

                        # if (nr_joc_template == '1_19' or nr_joc_template == '1_20') and (lines == 200 and columns == 0):
                        #     are_cercuri = detecteaza_cercuri(patch.copy(), show=nr_joc_template, lines=lines, columns=columns)[0]

                        if are_cercuri is not None:
                            img_despartita = patch[0:100, 0:100]
                            img_despartita_2 = patch[100:200, 0:200]
                            are_cercuri_1, nr_cercuri_1 = detecteaza_cercuri(img_despartita)
                            are_cercuri_2, nr_cercuri_2 = detecteaza_cercuri(img_despartita_2)
                            if (are_cercuri_1 is not None) and (are_cercuri_2 is not None):
                                line_1, col_1 = translate_line_column(lines, columns)
                                line_2, col_2 = translate_line_column(lines + 100, columns)

                                if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii < 2):
                                    nr_solutii = nr_solutii + 1
                                    piese_tabla.append((nr_joc, line_1, col_1))
                                    piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                    str_tuplu = str(line_1) + str(col_1) + " " + str(nr_cercuri_1) + str("\n")
                                    fisier.writelines(str_tuplu)
                                    print("Linie si col", line_1, col_1)

                                if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                    nr_solutii = nr_solutii + 1
                                    piese_tabla.append((nr_joc, line_2, col_2))
                                    piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                    str_tuplu2 = str(line_2) + str(col_2) + " " + str(nr_cercuri_2) + str("\n")
                                    fisier.writelines(str_tuplu2)
                                    print("Linie si col", line_2, col_2)

                            if (are_cercuri_1 is None) or (are_cercuri_2 is None):
                                low_yellow_fara_cercuri = (90, 0, 190)
                                high_yellow_fara_cercuri = (255, 100, 255)

                                img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                total_pixels = mask_yellow_hsv_img_despartita_1.size
                                percentage_white = (white_pixels / total_pixels) * 100

                                if (are_cercuri_1 is None) and (percentage_white >= 70):
                                    line_1, col_1 = translate_line_column(lines, columns)

                                    if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii < 2):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_1, col_1))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                        str_tuplu = str(line_1) + str(col_1) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_1, col_1)

                                    img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                    mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                                  low_yellow_fara_cercuri,
                                                                                  high_yellow_fara_cercuri)

                                    white_pixels = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                    total_pixels = mask_yellow_hsv_img_despartita_2.size
                                    percentage_white = (white_pixels / total_pixels) * 100

                                    if (are_cercuri_2 is not None) and (percentage_white >= 60):
                                        line_2, col_2 = translate_line_column(lines + 100, columns)

                                        if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                            nr_solutii = nr_solutii + 1
                                            piese_tabla.append((nr_joc, line_2, col_2))
                                            piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                            str_tuplu2 = str(line_2) + str(col_2) + " " + str(nr_cercuri_2) + str("\n")
                                            fisier.writelines(str_tuplu2)
                                            print("Linie si col", line_2, col_2)

                                img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                total_pixels = mask_yellow_hsv_img_despartita_2.size
                                percentage_white = (white_pixels / total_pixels) * 100

                                if (are_cercuri_2 is None) and (percentage_white >= 70):
                                    line_2, col_2 = translate_line_column(lines + 100, columns)

                                    if ((nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii < 2):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_2, col_2))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                        str_tuplu = str(line_2) + str(col_2) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_2, col_2)

                                    img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                    mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                                  low_yellow_fara_cercuri,
                                                                                  high_yellow_fara_cercuri)

                                    white_pixels = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                    total_pixels = mask_yellow_hsv_img_despartita_1.size
                                    percentage_white = (white_pixels / total_pixels) * 100

                                    if (are_cercuri_1 is not None) and (percentage_white >= 60):
                                        line_1, col_1 = translate_line_column(lines, columns)

                                        if ((nr_joc, line_1, col_1) not in piese_tabla) and (nr_solutii <= 2):
                                            nr_solutii = nr_solutii + 1
                                            piese_tabla.append((nr_joc, line_1, col_1))
                                            piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                            str_tuplu2 = str(line_1) + str(col_1) + " " + str(nr_cercuri_1) + str("\n")
                                            fisier.writelines(str_tuplu2)
                                            print("Linie si col", line_1, col_1)

                # piese pe orizontala pentru domino cu 0 cercuri
                for lines in range(0, 1500, 100):
                    for columns in range(i, 1500 if i != 0 else 1300, 200):
                        patch = result[lines:lines + 100, columns:columns + 200].copy()
                        are_cercuri = detecteaza_cercuri(patch.copy())[0]

                        if are_cercuri is None:
                            img_despartita = patch[0:100, 0:100]
                            img_despartita_2 = patch[0:100, 100:200]

                            are_cercuri_1, nr_cercuri_1 = detecteaza_cercuri(img_despartita)
                            are_cercuri_2, nr_cercuri_2 = detecteaza_cercuri(img_despartita_2)
                            if (are_cercuri_1 is None) and (are_cercuri_2 is None):
                                low_yellow_fara_cercuri = (90, 0, 190)
                                high_yellow_fara_cercuri = (255, 100, 255)

                                img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels_1 = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                total_pixels_1 = mask_yellow_hsv_img_despartita_1.size
                                percentage_white_1 = (white_pixels_1 / total_pixels_1) * 100

                                img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels_2 = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                total_pixels_2 = mask_yellow_hsv_img_despartita_2.size
                                percentage_white_2 = (white_pixels_2 / total_pixels_2) * 100

                                if (are_cercuri_1 is None) and (percentage_white_1 >= 70) and (
                                        are_cercuri_2 is None) and (percentage_white_2 >= 70):
                                    line_1, col_1 = translate_line_column(lines, columns)
                                    line_2, col_2 = translate_line_column(lines, columns + 100)

                                    if ((nr_joc, line_1, col_1) not in piese_tabla) and (
                                            (nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii == 0):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_1, col_1))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                        str_tuplu = str(line_1) + str(col_1) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_1, col_1)

                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_2, col_2))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                        str_tuplu2 = str(line_2) + str(col_2) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu2)
                                        print("Linie si col", line_2, col_2)

                # piese pe verticala
                for lines in range(i, 1500 if i != 0 else 1300, 200):
                    for columns in range(0, 1500, 100):
                        patch = result[lines:lines + 200, columns:columns + 100].copy()

                        are_cercuri = detecteaza_cercuri(patch.copy())[0]

                        if are_cercuri is None:
                            img_despartita = patch[0:100, 0:100]
                            img_despartita_2 = patch[100:200, 0:200]

                            are_cercuri_1, nr_cercuri_1 = detecteaza_cercuri(img_despartita)
                            are_cercuri_2, nr_cercuri_2 = detecteaza_cercuri(img_despartita_2)
                            if (are_cercuri_1 is None) and (are_cercuri_2 is None):
                                low_yellow_fara_cercuri = (90, 0, 190)
                                high_yellow_fara_cercuri = (255, 100, 255)

                                img_hsv_fara_cercuri_1 = cv.cvtColor(img_despartita.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_1 = cv.inRange(img_hsv_fara_cercuri_1,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels_1 = np.sum(mask_yellow_hsv_img_despartita_1 >= 200)
                                total_pixels_1 = mask_yellow_hsv_img_despartita_1.size
                                percentage_white_1 = (white_pixels_1 / total_pixels_1) * 100

                                img_hsv_fara_cercuri_2 = cv.cvtColor(img_despartita_2.copy(), cv.COLOR_BGR2HSV)
                                mask_yellow_hsv_img_despartita_2 = cv.inRange(img_hsv_fara_cercuri_2,
                                                                              low_yellow_fara_cercuri,
                                                                              high_yellow_fara_cercuri)

                                white_pixels_2 = np.sum(mask_yellow_hsv_img_despartita_2 >= 200)
                                total_pixels_2 = mask_yellow_hsv_img_despartita_2.size
                                percentage_white_2 = (white_pixels_2 / total_pixels_2) * 100

                                if (are_cercuri_1 is None) and (percentage_white_1 >= 70) and (
                                        are_cercuri_2 is None) and (percentage_white_2 >= 70):
                                    line_1, col_1 = translate_line_column(lines, columns)
                                    line_2, col_2 = translate_line_column(lines + 100, columns)

                                    if ((nr_joc, line_1, col_1) not in piese_tabla) and (
                                            (nr_joc, line_2, col_2) not in piese_tabla) and (nr_solutii == 0):
                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_1, col_1))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_1, col_1))
                                        str_tuplu = str(line_1) + str(col_1) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu)
                                        print("Linie si col", line_1, col_1)

                                        nr_solutii = nr_solutii + 1
                                        piese_tabla.append((nr_joc, line_2, col_2))
                                        piese_careu_punctaj.append((nr_joc, nr_mutare, line_2, col_2))
                                        str_tuplu2 = str(line_2) + str(col_2) + " " + str(0) + str("\n")
                                        fisier.writelines(str_tuplu2)
                                        print("Linie si col", line_2, col_2)

            if nr_solutii < 2:
                print(f"Nu am gasit solutii pentru {nr_joc_template}, scriem default in fisier")
                if nr_solutii == 0:
                    piese_tabla.append((nr_joc, 8, 'H'))
                    piese_tabla.append((nr_joc, 8, 'G'))
                    fisier.writelines("8H 6")
                    fisier.writelines("8G 6")
                elif nr_solutii == 1:
                    piese_tabla.append((nr_joc, 8, 'H'))
                    fisier.writelines("8H 6")

            fisier.close()

            fisier_pct_read = open(rezultat_path, 'r')
            lines_pct = fisier_pct_read.readlines()
            punctaj = 0

            linie_mutare1 = ''.join(map(str, [int(i) for i in lines_pct[0].strip().split(" ")[0] if i.isdigit()]))
            linie_mutare2 = ''.join(map(str, [int(i) for i in lines_pct[1].strip().split(" ")[0] if i.isdigit()]))
            col_mutare1 = ''.join(map(str, [str(i) for i in lines_pct[0].strip().split(" ")[0] if not i.isdigit()]))
            col_mutare2 = ''.join(map(str, [str(i) for i in lines_pct[1].strip().split(" ")[0] if not i.isdigit()]))
            l_pct = lines_pct[0].strip().split(" ")[1]
            c_pct = lines_pct[1].strip().split(" ")[1]

            line1_pct, column1_pct = careu_pct_line_column(int(linie_mutare1), col_mutare1)
            line2_pct, column2_pct = careu_pct_line_column(int(linie_mutare2), col_mutare2)

            careu_pct_piesa_1 = punctaj_careu[line1_pct][column1_pct]
            if careu_pct_piesa_1 != 0:
                print(
                    f"Avem piesa {linie_mutare1}{col_mutare1} care a fost mutata pe casuta cu punct. Punct casuta: {careu_pct_piesa_1} in jocul {nr_joc_template}.")

                if l_pct == c_pct:
                    punctaj = punctaj + 2 * careu_pct_piesa_1
                else:
                    punctaj = punctaj + careu_pct_piesa_1

            careu_pct_piesa_2 = punctaj_careu[line2_pct][column2_pct]
            if careu_pct_piesa_2 != 0:
                print(
                    f"Avem piesa {linie_mutare2}{col_mutare2} care a fost mutata pe casuta cu punct. Punct casuta: {careu_pct_piesa_2} in jocul {nr_joc_template}.")

                if l_pct == c_pct:
                    punctaj = punctaj + 2 * careu_pct_piesa_2
                else:
                    punctaj = punctaj + careu_pct_piesa_2

            fisier_pct_read.close()

            fisier_pct = open(rezultat_path, 'a')
            print(f"Scriem punctajul in fisier pentru {nr_joc_template}")
            fisier_pct.writelines(str(punctaj))

            fisier_pct.close()

            nr_mutare = nr_mutare + 1


def detecteaza_cercuri(img, show=0, lines=0, columns=0):
    # Convert to grayscale
    # show_image('img', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Use the Hough Circle Transform to detect circles
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,  # Inverse ratio of the accumulator resolution to the image resolution (1 means the same resolution)
        minDist=25,  # Minimum distance between the centers of detected circles
        param1=600,  # Higher threshold for the internal Canny edge detector
        param2=25,  # Threshold for circle detection (lower means more circles will be detected)
        minRadius=10,  # Minimum radius of the detected circles
        maxRadius=14  # Maximum radius of the detected circles
    )

    if show != 0:
        # If circles are found, draw them on the image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 6)
                # Draw the center of the circle
                cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

            print("nr=", circles.shape[1])
            print("lines,columns=", lines, " ", columns)
            show_image('blurred', blurred)
            show_image('cerc', img)

    if circles is not None:
        num_circles = circles.shape[1]
        return img, num_circles
    else:
        return None, None


show_images()
