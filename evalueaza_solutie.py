#to be written
import os

punctaj_final = 2

files = os.listdir('antrenare')
nr_joc = 1
nr_mutare = 1
for file in files:
    if nr_mutare == 21:
        nr_joc = nr_joc + 1
        nr_mutare = 1
    if nr_joc == 6:
        break
    if file[-3:] == 'txt':
        nr_joc_template = str(nr_joc) + '_' + ('0' if nr_mutare <= 9 else '') + str(nr_mutare)
        image_path = 'antrenare/' + nr_joc_template + '.txt'
        image_path_my_sol = 'rezultate/' + nr_joc_template + '.txt'

        f = open(image_path)
        f_my_sol = open(image_path_my_sol)
        for line in f.readlines():
            for line_my_sol in f_my_sol.readlines():
                original_position = line.split(' ')
                my_position = line_my_sol.split(' ')
                if original_position[0] == my_position[0]:
                    punctaj_final = punctaj_final + 0.05

                if original_position[1] == my_position[1]:
                    punctaj_final = punctaj_final + 0.02

    nr_mutare = nr_mutare + 1

print(punctaj_final)