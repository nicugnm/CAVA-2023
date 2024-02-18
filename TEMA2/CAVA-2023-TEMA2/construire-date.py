import cv2 as cv
import os
import numpy as np

character_path = lambda name: f'antrenare/{name}/'

annotation_path = lambda name: f'antrenare/{name}_annotations.txt'

positive_example = lambda name: f'solutie_personala/data/exemple_pozitive/{name}/'
negative_example = lambda name: f'solutie_personala/data/exemple_negative/{name}/'
solutii_totale_pozitive = 'solutie_personala/data/exemple_pozitive_totale/'
solutii_totale_negative = 'solutie_personala/data/exemple_negative_totale/'

def show_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def exemple_pozitive(name):
    files = os.listdir(character_path(name))
    f = open(annotation_path(name), "r")
    lines_annotations = f.readlines()
    for file in files:
        if (file[-3:] == "jpg"):
            img = cv.imread(character_path(name) + file)
            # show_image('img', img)
            number_file = int(file.replace('.jpg', ''))
            for idx, line_annotation in enumerate(lines_annotations):
                current_annotation = lines_annotations[idx].split()
                if current_annotation[0] == file and current_annotation[5] == name:
                    x_min = int(current_annotation[1])
                    y_min = int(current_annotation[2])
                    x_max = int(current_annotation[3])
                    y_max = int(current_annotation[4])

                    patch = img[y_min:y_max, x_min:x_max]
                    patch = cv.resize(patch, (72, 72))
                    print(f'Exemplu pozitiv pentru {name} gasit: ' + f'{name}_' + str(number_file) + ".jpg")
                    cv.imwrite(positive_example(name) + f"{name}_" + str(number_file) + ".jpg", patch)
                    cv.imwrite(solutii_totale_pozitive + f"{name}_" + str(number_file) + ".jpg", patch)


def exemple_pozitive_unknown():
    directories = ['barney', 'betty', 'fred', 'wilma']
    for name in directories:
        files = os.listdir(character_path(name))
        f = open(annotation_path(name), "r")
        lines_annotations = f.readlines()
        name_unknown = 'unknown'
        for file in files:
            if (file[-3:] == "jpg"):
                img = cv.imread(character_path(name) + file)
                # show_image('img', img)
                number_file = int(file.replace('.jpg', ''))
                for idx, line_annotation in enumerate(lines_annotations):
                    current_annotation = lines_annotations[idx].split()
                    if current_annotation[0] == file and current_annotation[5] == name_unknown:
                        x_min = int(current_annotation[1])
                        y_min = int(current_annotation[2])
                        x_max = int(current_annotation[3])
                        y_max = int(current_annotation[4])

                        patch = img[y_min:y_max, x_min:x_max]
                        patch = cv.resize(patch, (72, 72))
                        print(f'Exemplu pozitiv pentru {name_unknown} gasit: ' + f'{name_unknown}_' + name + '_' + str(
                            number_file) + ".jpg")
                        cv.imwrite(positive_example(name_unknown) + f"{name_unknown}_" + name + '_' + str(
                            number_file) + ".jpg", patch)
                        cv.imwrite(solutii_totale_negative + f"{name_unknown}_" + name + '_' + str(
                            number_file) + ".jpg", patch)


def salvare_adnotari(name):
    f = open(annotation_path(name), "r")
    annotations_dictionary = {}
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        if line[0] not in annotations_dictionary.keys():
            annotations_dictionary[line[0]] = []
            annotations_dictionary[line[0]].append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])
        else:
            annotations_dictionary[line[0]].append([int(line[1]), int(line[2]), int(line[3]), int(line[4])])

    return annotations_dictionary


def overlap(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    return not (y1_min > y2_max or y2_min > y1_max) or not (x1_max > x2_min or x2_max > x1_min)


def exemple_negative(name, character_path_exemple_negative, image_folder):
    annotations_dictionary = salvare_adnotari(name)
    files = os.listdir(image_folder)
    idx = 0
    for file in files:
        if file[-3:] == "jpg":
            img = cv.imread(image_folder + file)
            nr_rows = img.shape[1]
            nr_cols = img.shape[0]
            finish = False
            idx += 1
            time_spent = 0
            while not finish:
                x_min = np.random.randint(0, nr_rows - 75)
                y_min = np.random.randint(0, nr_cols - 75)
                x_max = x_min + 72
                y_max = y_min + 72
                time_spent += 1
                number_of_faces = 0
                for list in annotations_dictionary[file]:
                    if not overlap(-x_min, y_min, -x_max, y_max, -list[0], list[1], -list[2], list[3]):
                        number_of_faces += 1
                if number_of_faces == len(annotations_dictionary[file]):
                    print(y_min, y_max, x_min, x_max)
                    patch = img[y_min:y_max, x_min:x_max]
                    patch = cv.resize(patch, (72, 72))
                    cv.imwrite(character_path_exemple_negative + f"{name}_" + "negative" + str(idx) + ".jpg", patch)
                    cv.imwrite(solutii_totale_negative + f"{name}_" + "negative" + str(idx) + ".jpg", patch)

                    finish = True
                if time_spent > 100_000:
                    x_min = np.random.randint(0, nr_rows - 22)
                    y_min = np.random.randint(0, nr_cols - 22)
                    x_max = x_min + 20
                    y_max = y_min + 20
                    print(time_spent)
                if time_spent > 500_000:
                    break


exemple_pozitive('barney')
exemple_pozitive('betty')
exemple_pozitive('fred')
exemple_pozitive('wilma')
exemple_pozitive_unknown()

exemple_negative("barney", negative_example("barney"), character_path("barney"))
exemple_negative("betty", negative_example("betty"), character_path("betty"))
exemple_negative("fred", negative_example("fred"), character_path("fred"))
exemple_negative("wilma", negative_example("wilma"), character_path("wilma"))
