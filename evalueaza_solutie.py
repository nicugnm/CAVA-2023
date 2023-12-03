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
        lines = f.readlines()
        my_lines = f_my_sol.readlines()

        if len(my_lines) > 2:
            print("Am gasit mai mult de 2 mutari la:", nr_joc_template)

        first_element_original = lines[0].strip()
        second_element_original = lines[1].strip()

        mine_first_element = my_lines[0].strip()
        mine_second_element = my_lines[1].strip()

        split_f_original = first_element_original.split(" ")
        split_s_original = second_element_original.split(" ")

        split_f_mine = mine_first_element.split(" ")
        split_s_mine = mine_second_element.split(" ")

        if ((split_f_original[0] == split_f_mine[0]) or (split_f_original[0] == split_s_mine[0])) and ((split_s_original[0] == split_f_mine[0]) or (split_s_original[0] == split_s_mine[0])):
            if (split_f_original[0] == split_f_mine[0]) and (split_s_original[0] == split_f_mine[0]):
                if split_f_original[1] == split_f_mine[1] and split_s_original[1] == split_f_mine[1]:
                    punctaj_final = punctaj_final + 0.02
                else:
                    print("diferenta pct=", nr_joc_template)
            elif (split_f_original[0] == split_f_mine[0]) and (split_s_original[0] == split_s_mine[0]):
                if split_f_original[1] == split_f_mine[1] and split_s_original[1] == split_s_mine[1]:
                    punctaj_final = punctaj_final + 0.02
                else:
                    print("diferenta pct=", nr_joc_template)

            elif (split_f_original[0] == split_s_mine[0]) and (split_s_original[0] == split_f_mine[0]):
                if split_f_original[1] == split_s_mine[1] and split_s_original[1] == split_f_mine[1]:
                    punctaj_final = punctaj_final + 0.02
                else:
                    print("diferenta pct=", nr_joc_template)
            elif (split_f_original[0] == split_s_mine[0]) and (split_s_original[0] == split_s_mine[0]):
                if split_f_original[1] == split_s_mine[1] and split_s_original[1] == split_s_mine[1]:
                    punctaj_final = punctaj_final + 0.02
                else:
                    print("diferenta pct=", nr_joc_template)
            else:
                print("diferenta pct=", nr_joc_template)

            punctaj_final = punctaj_final + 0.05
        else:
            print("diferenta piesa=",nr_joc_template)

        nr_mutare = nr_mutare + 1

print("pct final=",punctaj_final,"/9")