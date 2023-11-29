import cv2 as cv
import numpy as np
import os

# mai jos sunt pathurile ce trebuie inlocuite pe fiecare masina cu path-ul potrivit
# atentie la separatorul \ din path deoarece unele combinatii \t sau \n pot 
# face ca pathul sa fie invalid, atunci cand se intampla asta, doar inlocuiti \ cu \\
# de asemenea path-ul trebuie sa mai aiba la final un \\ ca programul sa functioneze cum trebuie

# pathul unde sunt stocate solutiile ce vor fi testate in evalueaza_solutie.py
path_solutii = "antrenare"

# pathul in care se afla imaginiile de antrenare
path_antrenare = "antrenare"

# pathul unde sunt stocate templates-urile oferite in folder-ul templates
path_templates = "antrenare"


def show_image(title, image):
    # image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(image):
    image = cv.resize(image,(0,0),fx=0.2,fy=0.2)
    image = image[240:680,90:555]
    low_yellow = (0, 0, 0)
    high_yellow = (255, 140, 255)
 
    img_hsv = cv.cvtColor(image.copy(), cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    contours, _ = cv.findContours(mask_yellow_hsv,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            possible_bottom_left = None
            possible_top_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point
                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point
                
                
            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 15*40
    height = 15*40
    
    image_copy = image.copy()
    cv.circle(image_copy,tuple(top_left),10,(0,0,255),-1) #red
    cv.circle(image_copy,tuple(top_right),10,(0,255,255),-1) #yellow
    cv.circle(image_copy,tuple(bottom_left),10,(255,0,255),-1)#pink
    cv.circle(image_copy,tuple(bottom_right),10,(255,255,255),-1)#white
    # show_image("detected corners",image_copy)

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    #result = cv.cvtColor(result,cv.COLOR_GRAY2BGR)
    return result

lines_horizontal=[]
for i in range(0,601,40):
    l=[]
    l.append((0,i))
    l.append((599,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,601,40):
    l=[]
    l.append((i,0))
    l.append((i,599))
    lines_vertical.append(l)

def determina_configuratie_careu_ox(img_hsv,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='str')
    show_image("mask_hsv",img_hsv)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]+5
            y_max = lines_vertical[j + 1][1][0]-5
            x_min = lines_horizontal[i][0][1]+5
            x_max = lines_horizontal[i + 1][1][1]-5
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                matrix[i][j]='x'
            else:
                matrix[i][j]='o'
    return matrix

def vizualizare_configuratie(result,matrix,lines_horizontal,lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == 'x': 
                cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)

def clasifica_litera(patch,path_temp):
        maxi=-np.inf
        litera=""
        # show_image("patch",patch)
        patch = cv.cvtColor(patch.copy(),cv.COLOR_BGR2GRAY)
        files = os.listdir(path_temp)
        for file in files:
            img_template = cv.imread(path_temp+file)
            img_template= cv.cvtColor(img_template,cv.COLOR_BGR2GRAY)
            # show_image("temp",img_template)
            corr = cv.matchTemplate(patch,img_template,  cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)
            if corr>maxi :
                maxi=corr
                litera=file[0]
        return litera

def memoreaza_templates(img_hsv,img,path_temp,nr_exetension =""):
    index = 0
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            nr = 4
            y_min = lines_vertical[j][0][0]+nr
            y_max = lines_vertical[j + 1][1][0]-nr
            x_min = lines_horizontal[i][0][1]+nr
            x_max = lines_horizontal[i + 1][1][1]-nr
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            y_min = lines_vertical[j][0][0]+2*nr
            y_max = lines_vertical[j + 1][1][0]-2*nr
            x_min = lines_horizontal[i][0][1]+2*nr
            x_max = lines_horizontal[i + 1][1][1]-2*nr
            patch_org = img[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                litera_curenta=clasifica_litera(patch_org,path_templates)
                filename = litera_curenta+nr_exetension+str(index)+".jpg"
                index +=1
                cv.imwrite(path_temp+filename,patch_org)
    return



def determina_configuratie_careu_olitere(img_hsv,lines_horizontal,lines_vertical,img_original,path_temp):
    matrix = np.empty((15,15), dtype='str')
    show_image("mask_hsv",img_hsv)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]+5
            y_max = lines_vertical[j + 1][1][0]-5
            x_min = lines_horizontal[i][0][1]+5
            x_max = lines_horizontal[i + 1][1][1]-5
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            y_min = lines_vertical[j][0][0]+8
            y_max = lines_vertical[j + 1][1][0]-8
            x_min = lines_horizontal[i][0][1]+8
            x_max = lines_horizontal[i + 1][1][1]-8
            patch_original = img_original[x_min:x_max, y_min:y_max].copy()
            show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                matrix[i][j]=clasifica_litera(patch_original,path_temp)
            else:
                matrix[i][j]="o"
    return matrix



matrice_punctaje =[
    [5,1,1,2,1,1,1,5,1,1,1,2,1,1,5],
    [1,4,1,1,1,3,1,1,1,3,1,1,1,4,1],
    [1,1,4,1,1,1,2,1,2,1,1,1,4,1,1],
    [2,1,1,4,1,1,1,2,1,1,1,4,1,1,2],
    [1,1,1,1,4,1,1,1,1,1,4,1,1,1,1],
    [1,3,1,1,1,3,1,1,1,3,1,1,1,3,1],
    [1,1,2,1,1,1,2,1,2,1,1,1,2,1,1],
    [5,1,1,2,1,1,1,4,1,1,1,2,1,1,5],
    [1,1,2,1,1,1,2,1,2,1,1,1,2,1,1],
    [1,3,1,1,1,3,1,1,1,3,1,1,1,3,1],
    [1,1,1,1,4,1,1,1,1,1,4,1,1,1,1],
    [2,1,1,4,1,1,1,2,1,1,1,4,1,1,2],
    [1,1,4,1,1,1,2,1,2,1,1,1,4,1,1],
    [1,4,1,1,1,3,1,1,1,3,1,1,1,4,1],
    [5,1,1,2,1,1,1,5,1,1,1,2,1,1,5],
]

dictionar_poz_litera = {}
for i in range(15):
    dictionar_poz_litera[i+1]=chr(ord('A')+i)

dictionar_punctaj_litera={}
dictionar_punctaj_litera["A"]=1
dictionar_punctaj_litera["B"]=9
dictionar_punctaj_litera["C"]=1
dictionar_punctaj_litera["D"]=2
dictionar_punctaj_litera["E"]=1
dictionar_punctaj_litera["F"]=8
dictionar_punctaj_litera["G"]=9
dictionar_punctaj_litera["H"]=10
dictionar_punctaj_litera["I"]=1
dictionar_punctaj_litera["J"]=10
dictionar_punctaj_litera["L"]=1
dictionar_punctaj_litera["M"]=4
dictionar_punctaj_litera["N"]=1
dictionar_punctaj_litera["O"]=1
dictionar_punctaj_litera["P"]=2
dictionar_punctaj_litera["R"]=1
dictionar_punctaj_litera["S"]=1
dictionar_punctaj_litera["T"]=1
dictionar_punctaj_litera["U"]=1
dictionar_punctaj_litera["V"]=8
dictionar_punctaj_litera["X"]=10
dictionar_punctaj_litera["Z"]=10
dictionar_punctaj_litera["0"]=0

def calculeaza_cuv_verticala(matrice_curenta,matrice_anterioara,pozitii):
    poz_prima_litera = pozitii
    # parcurgem pe verticala pana la capatul superior
    index_1 = poz_prima_litera[0]
    # print("index before while is "+matrice_curenta[index_1][poz_prima_litera[1]])
    while(matrice_curenta[index_1][poz_prima_litera[1]]!='o' and index_1>=0):
        index_1-=1
    index_1+=1
    scor_cuv_verticala = 0
    triplare_cuvant = 0
    dublare_cuvant=0
    lungime_cuv = 0
    # print("index is "+str(index_1))
    # incepem sa cream cuvantul pe verticala
    while(index_1<=14 and matrice_curenta[index_1][poz_prima_litera[1]]!='o'):
        if(lungime_cuv==0 and index_1==14):
            break
        lungime_cuv+=1
        poz_1 = index_1
        poz_2 = poz_prima_litera[1]
        litera_curenta = matrice_curenta[poz_1][poz_2]
        if(matrice_anterioara[poz_1][poz_2]=='o'):
            if(matrice_punctaje[poz_1][poz_2]==5):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
                triplare_cuvant+=1
            if(matrice_punctaje[poz_1][poz_2]==4):
                dublare_cuvant+=1
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
            if(matrice_punctaje[poz_1][poz_2]==3):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]*3
            if(matrice_punctaje[poz_1][poz_2]==2):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]*2
            if(matrice_punctaje[poz_1][poz_2]==1):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
        else:
            scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
        index_1+=1
    scor_cuv_verticala = scor_cuv_verticala*(2**dublare_cuvant)
    scor_cuv_verticala = scor_cuv_verticala*(3**triplare_cuvant)
    # print("scor cuvant pe verticala cand principala e orizontala = "+str(scor_cuv_verticala)+"\n")
    if(lungime_cuv>1):
        return scor_cuv_verticala
    return 0

def calculeaza_cuv_orizontala(matrice_curenta,matrice_anterioara,pozitii):
        # parcurgem pe orizontala pana la capatul din stanga al cuvantului
    poz_prima_litera = pozitii
    index_1 = poz_prima_litera[1]
    while(matrice_curenta[poz_prima_litera[0]][index_1]!='o' and index_1>=0):
        index_1-=1
    index_1+=1
    scor_cuv_orizontala = 0
    triplare_cuvant = 0
    dublare_cuvant=0
    lungime_cuv=0
    # incepem sa cream cuvantul pe orizontala
    while(index_1<=14 and matrice_curenta[poz_prima_litera[0]][index_1]!='o'):
        if(lungime_cuv == 0 and index_1==14):
            break
        lungime_cuv+=1
        poz_1 = poz_prima_litera[0]
        poz_2 = index_1
        litera_curenta = matrice_curenta[poz_1][poz_2]
        if(matrice_anterioara[poz_1][poz_2]=='o'):
            if(matrice_punctaje[poz_1][poz_2]==5):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
                triplare_cuvant+=1
            if(matrice_punctaje[poz_1][poz_2]==4):
                dublare_cuvant+=1
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
            if(matrice_punctaje[poz_1][poz_2]==3):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]*3
            if(matrice_punctaje[poz_1][poz_2]==2):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]*2
            if(matrice_punctaje[poz_1][poz_2]==1):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
        else:
            scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
        index_1+=1
    scor_cuv_orizontala = scor_cuv_orizontala*(2**dublare_cuvant)
    scor_cuv_orizontala = scor_cuv_orizontala*(3**triplare_cuvant)
    # print("index = "+str(matrice_curenta))
    # print("scor cuvant pe orizontala cand verticala e principala = "+str(scor_cuv_orizontala)+"\n")
    if(lungime_cuv>1):
        return scor_cuv_orizontala
    return 0

def obtine_scor_runda(matrice_curenta,matrice_anterioara):
    scor_total = 0
    lista_poz_litere_noi=[]
    for i in range(15):
        for j in range(15):
            if matrice_anterioara[i][j]=='o' and matrice_curenta[i][j]!='o':
                lista_poz_litere_noi.append((i,j))

    cuvant_pe_verticala = False
    cuvant_pe_orizontala = False
    poz_prima_litera = lista_poz_litere_noi[0]
    if(len(lista_poz_litere_noi)>=7):
        scor_total+=50
    # parcurgem pe verticala pana la capatul superior
    index_1 = poz_prima_litera[0]
    while(matrice_curenta[index_1][poz_prima_litera[1]]!='o' and index_1>=0):
        index_1-=1
    index_1+=1
    scor_cuv_verticala = 0
    triplare_cuvant = 0
    dublare_cuvant=0
    lungime_cuv = 0
    # incepem sa cream cuvantul pe verticala
    while(index_1<=14 and matrice_curenta[index_1][poz_prima_litera[1]]!='o'):
        if(lungime_cuv==0 and index_1==14):
            break
        lungime_cuv+=1
        poz_1 = index_1
        poz_2 = poz_prima_litera[1]
        litera_curenta = matrice_curenta[poz_1][poz_2]
        if(matrice_anterioara[poz_1][poz_2]=='o'):
            if(poz_1 != poz_prima_litera[0]):
                cuvant_pe_verticala = True
            if(matrice_punctaje[poz_1][poz_2]==5):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
                triplare_cuvant+=1
            if(matrice_punctaje[poz_1][poz_2]==4):
                dublare_cuvant+=1
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
            if(matrice_punctaje[poz_1][poz_2]==3):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]*3
            if(matrice_punctaje[poz_1][poz_2]==2):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]*2
            if(matrice_punctaje[poz_1][poz_2]==1):
                scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
        else:
            scor_cuv_verticala=scor_cuv_verticala+dictionar_punctaj_litera[litera_curenta]
        index_1+=1
    scor_cuv_verticala = scor_cuv_verticala*(2**dublare_cuvant)
    scor_cuv_verticala = scor_cuv_verticala*(3**triplare_cuvant)
    # print("scor cuvant pe verticala = "+str(scor_cuv_verticala)+"\n")
    if(lungime_cuv>1):
        scor_total = scor_total+scor_cuv_verticala

    # parcurgem pe orizontala pana la capatul din stanga al cuvantului
    index_1 = poz_prima_litera[1]
    while(matrice_curenta[poz_prima_litera[0]][index_1]!='o' and index_1>=0):
        index_1-=1
    index_1+=1
    scor_cuv_orizontala = 0
    triplare_cuvant = 0
    dublare_cuvant=0
    lungime_cuv=0
    # incepem sa cream cuvantul pe orizontala
    while(index_1<=14 and matrice_curenta[poz_prima_litera[0]][index_1]!='o'):
        if(lungime_cuv==0 and index_1==14):
            break
        lungime_cuv+=1
        poz_1 = poz_prima_litera[0]
        poz_2 = index_1
        litera_curenta = matrice_curenta[poz_1][poz_2]
        if(matrice_anterioara[poz_1][poz_2]=='o'):
            if(poz_2 != poz_prima_litera[1]):
                cuvant_pe_orizontala = True
            if(matrice_punctaje[poz_1][poz_2]==5):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
                triplare_cuvant+=1
            if(matrice_punctaje[poz_1][poz_2]==4):
                dublare_cuvant+=1
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
            if(matrice_punctaje[poz_1][poz_2]==3):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]*3
            if(matrice_punctaje[poz_1][poz_2]==2):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]*2
            if(matrice_punctaje[poz_1][poz_2]==1):
                scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
        else:
            scor_cuv_orizontala=scor_cuv_orizontala+dictionar_punctaj_litera[litera_curenta]
        index_1+=1
    scor_cuv_orizontala = scor_cuv_orizontala*(2**dublare_cuvant)
    scor_cuv_orizontala = scor_cuv_orizontala*(3**triplare_cuvant)
    # print("dublare cuvant = "+str(dublare_cuvant))
    # print("scor cuvant pe orizontala = "+str(scor_cuv_orizontala)+"\n")
    if(lungime_cuv>1):
        scor_total = scor_total+scor_cuv_orizontala

    if(cuvant_pe_orizontala and cuvant_pe_verticala):
        print("error")
        exit(1)
    if(cuvant_pe_verticala):
        for pozitii in lista_poz_litere_noi[1:]:
            scor_total = scor_total+calculeaza_cuv_orizontala(matrice_curenta,matrice_anterioara,pozitii)
    else:
        for pozitii in lista_poz_litere_noi[1:]:
            scor_total = scor_total+calculeaza_cuv_verticala(matrice_curenta,matrice_anterioara,pozitii)
    return scor_total




matrice_initiala = np.empty((15,15),dtype="str")
for i in range(15):
    for j in range(15):
        matrice_initiala[i][j]='o'


def scrie_rezultate(path_train,path_sol,path_temp):
    matrice_viz = np.zeros((15,15),dtype=int)
    lista_rezultate = []
    files=os.listdir(path_train)
    game_nr= 1
    matrice_anterioara=matrice_initiala
    for file in files:
        if file[-3:]=='jpg':
            if(file[-6:] == "01.jpg"):
                matrice_viz = np.zeros((15,15),dtype=int)
                matrice_anterioara=matrice_initiala
                if(len(lista_rezultate)!=0):
                    for i in range(len(lista_rezultate)):
                        if i >=0 and i <=8:
                            file_writer = open(path_sol+str(game_nr)+"_0"+str(i+1)+".txt","w")
                        else:
                            file_writer = open(path_sol+str(game_nr)+"_"+str(i+1)+".txt","w")
                        for element in lista_rezultate[i]:
                            file_writer.write(element+"\n")
                        file_writer.close
                lista_rezultate = []
            lista_runda = []
            img = cv.imread(path_train+ "/" + file)
            result=extrage_careu(img)
            low_yellow = (0, 0, 239)
            high_yellow = (255, 111, 255)
            img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
            mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
            matrice=determina_configuratie_careu_olitere(mask_yellow_hsv,lines_horizontal,lines_vertical,result,path_temp)
            for i in range(15):
                for j in range(15):
                    if(matrice[i][j]!='o'):
                        if(matrice_viz[i][j]==0):
                            matrice_viz[i][j]=1
                            if matrice[i][j] == "0":
                                lista_runda.append(str(i+1)+dictionar_poz_litera[j+1]+" "+"?")
                            else:
                                lista_runda.append(str(i+1)+dictionar_poz_litera[j+1]+" "+matrice[i][j])
            scor_runda = obtine_scor_runda(matrice,matrice_anterioara)
            # print("scor final = "+str(scor_runda))
            lista_runda.append(str(scor_runda))
            lista_rezultate.append(lista_runda)
            game_nr = int(file[-8])
            matrice_anterioara=matrice
    for i in range(len(lista_rezultate)):
        if i >=0 and i <=8:
            file_writer = open(path_sol+str(game_nr)+"_0"+str(i+1)+".txt","w")
        else:
            file_writer = open(path_sol+str(game_nr)+"_"+str(i+1)+".txt","w")
        for element in lista_rezultate[i]:
            file_writer.write(element+"\n")
        file_writer.close

scrie_rezultate(path_antrenare,path_solutii,path_templates)
