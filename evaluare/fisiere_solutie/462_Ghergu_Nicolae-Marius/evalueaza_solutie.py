import os

### INSTRUCTIUNI:
# By default, acum se citesc datele din fisierul "antrenare/" -> de ex: "antrenare/1_01.jpg" samd. numarul maxim de jocuri este 5 si numarul maxim de mutari este 20 (inclusiv 5 si 20)
# Rezultatele vor face output in folder-ul "rezultate/".
folder_testare = "antrenare"
folder_rezultate = "rezultate"
numar_mutari_per_joc = 20
numar_jocuri = 5

punctaj_final = 2 # oficiul + documentatie

files = os.listdir(folder_testare)
nr_joc = 1
nr_mutare = 1
for file in files:
    if nr_mutare == numar_mutari_per_joc + 1:
        nr_joc = nr_joc + 1
        nr_mutare = 1
    if nr_joc == numar_jocuri + 1:
        break
    if file[-3:] == 'txt':
        nr_joc_template = str(nr_joc) + '_' + ('0' if nr_mutare <= 9 else '') + str(nr_mutare)
        image_path = f'{folder_testare}/' + nr_joc_template + '.txt'
        image_path_my_sol = f'{folder_rezultate}/' + nr_joc_template + '.txt'

        f = open(image_path)
        f_my_sol = open(image_path_my_sol)
        lines = f.readlines()
        my_lines = f_my_sol.readlines()

        # if len(my_lines) > 2:
        #     print("Am gasit mai mult de 2 mutari la:", nr_joc_template)

        first_element_original = lines[0].strip()
        second_element_original = lines[1].strip()
        third_element_original = lines[2].strip()

        mine_first_element = my_lines[0].strip()
        mine_second_element = my_lines[1].strip()
        mine_third_element = my_lines[2].strip()

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

        if third_element_original == mine_third_element:
            punctaj_final = punctaj_final + 0.02
        else:
            print(f"diferenta punctaj jucator la {nr_joc_template}")

        nr_mutare = nr_mutare + 1

print("pct final=",punctaj_final,"/11")