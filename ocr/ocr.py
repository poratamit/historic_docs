import cv2
import numpy as np
import os
from pathlib import Path
import xlsxwriter
import easyocr
import re
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
reader = easyocr.Reader(['de'])
from .conf import BASE1_CONF, BASE2_CONF
from utils import get_registered_back_and_front
import pytesseract


def image2excel(imgfile, folderName, workbook):

    # read your file
    file = imgfile
    img = cv2.imread(file, 0)

    # the base of the pic
    base = int(file.split('Base')[1].split('.')[0])

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # inverting the image
    img_bin = 255 - img_bin

    # countcol(width) of kernel as 100th of total width

    kernel_len = np.array(img).shape[1] // 100
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)

    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=4)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.4, horizontal_lines, 0.15, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=1)
    thresh, img_vh = cv2.threshold(img_vh, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Create list box to store all boxes in

    box = createBoxes(base)
    # Get position (x,y), width and height for every contour and show the contour on image
    for i in range(len(box)):
        x, y, w, h = box[i][0], box[i][1], box[i][2], box[i][3]
        image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.imwrite("tmp/boxs" + folderName + ".jpg", image)
    outer = []

    for i in range(len(box)):
        if (0 < i):
            s = ''
            y, x, w, h = box[i][0], box[i][1], box[i][2], box[i][3]
            finalimg = bitxor[x:x + h, y:y + w]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            border = cv2.copyMakeBorder(finalimg, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255])
            resizing = cv2.resize(border, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
            dilation = cv2.dilate(resizing, kernel, iterations=0)
            erosion = cv2.erode(dilation, kernel, iterations=1)

            out = reader.readtext(erosion)

            for k in range(len(out)):
                s += out[k][1] + ' '
            out = s
            out = out.replace(';', ':')
            outer.append(out)

        elif (i == 0):

            y, x, w, h = box[i][0], box[i][1], box[i][2], box[i][3]
            finalimg = bitnot[x:x + h, y:y + w]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            border = cv2.copyMakeBorder(finalimg, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255])
            resizing = cv2.resize(border, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
            dilation = cv2.dilate(resizing, kernel, iterations=0)
            erosion = cv2.erode(dilation, kernel, iterations=1)

            out = pytesseract.image_to_string(erosion, lang='deu', config=r'--oem 3 --psm 6')
            if (len(out) == 0):
                out = pytesseract.image_to_string(erosion, config=r'--psm 6')
            out = out.strip('\x0c')
            out = out.strip('\n')
            out = out.replace(';', ':')
            outer.append(out)

        else:
            outer.append('')

    arr = np.array(outer)

    csvArr = [''] * len(arr)
    for i in range(len(csvArr)):
        csvArr[i] = arr[i]
    # cell 5
    csvArr[5] = ['', '']
    if (len(arr[5].splitlines()) > 1):

        for i in range(len(arr[5].splitlines())):
            csvArr[5][i] = arr[5].splitlines()[i]
    else:
        csvArr[5][0] = arr[5]

    # cell 11
    csvArr[11] = ['', '']

    if (len(arr[11].split(' ')) > 1):
        tmp = arr[11].split(' ')
        the_word = process.extract('Herdstelle', tmp, limit=1)[0][0]
        the_word2 = process.extract('Durchschleusungs-Nr', tmp, limit=1)[0][0]
        tmp2 = re.split(the_word + '|' + the_word2, arr[11])

        csvArr[11][0] = tmp2[1]
        csvArr[11][1] = tmp2[2]

    for i in range(len(csvArr)):
        if (type(csvArr[i]) == list):
            for j in range(len(csvArr[i])):
                csvArr[i][j] = csvArr[i][j].strip('\n')

        else:
            csvArr[i] = csvArr[i].strip('\n')

    Merkunftsland = ["G.G", "Polen", "Russland", "UdSSR", "Rubland", "Russl.", "GG W Ost"]
    Volkszugehorigkelt = ["Dtsch", "Poln", "Polen", "pole", "VD", "Deutsch", "Deutscher", "Ungeklail", 'ukrainisch', ""]
    Merkunftsland2 = ["Polen", "Warthegau", "Galizien", "Russland", "UdSSR", "Wolhynien", "Deutschland", 'Rubland',
                      'Litauen', 'ostpreubeh', 'westr.', '"']

    maxScore = 0
    the_word = csvArr[4].lower()
    the_word = process.extract(the_word, Merkunftsland, limit=1)[0][0]
    csvArr[4] = the_word

    for i in [37, 42, 45, 47, 49, 51]:
        if (len(csvArr[i]) > 2):
            maxScore = 0
            the_word = csvArr[i].lower()

            for j in range(len(Volkszugehorigkelt)):
                score = fuzz.token_sort_ratio(Volkszugehorigkelt[j].lower(), the_word)

                if (score > maxScore):
                    maxScore = score
                    the_final_word = Volkszugehorigkelt[j]

            if (maxScore >= 40):
                csvArr[i] = the_final_word

    maxScore = 0

    for i in [38, 43, 46, 48, 50, 52]:
        if (len(csvArr[i]) > 2):
            maxScore = 0
            the_word = csvArr[i].lower()
            for j in range(len(Merkunftsland2)):
                score = fuzz.token_sort_ratio(Merkunftsland2[j].lower(), the_word)
                if (score > maxScore):
                    maxScore = score
                    the_final_word = Merkunftsland2[j]
            if (maxScore >= 40):
                csvArr[i] = the_final_word

    for i in [39, 40]:
        if (len(csvArr[i]) > 0):
            csvArr[i] = re.sub('\D', '', csvArr[i])

    print(folderName)
    print(csvArr)
    createCsv(csvArr, folderName, workbook)


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def createCsv(arr, folderName, workbook):
    list_Of_Elments = ["Name", "Vorname", "Durchschleusungs-Nr", "geborene", "herkunftsland",
                       ["verwitwete", "geschiedene"], "Kinder ohen Rus-Karte", ["Herdstelle", "Durchschleusungs-Nr"],
                       "Geburtstag", "Geburtsort", "Wohnort", "Fam-Stand", "Heiratsjahr", "Kinder", "davon leben",
                       "im 1.Lebens-jahr gestorben", "Geschwister", "davon leben", "Religion", "Beruf", "Gedient",
                       "Volkszugehorigkeit", "Herkunftsland", "KorpergroBe", "Gewicht", "Brillentrager"]
    marks = ['v', 'vv', 'vm', 'm', 'mv', 'mm']
    folderName = folderName.split('-Base')[0].split('R 9361')[1]
    worksheet = workbook.add_worksheet(folderName)
    worksheet.set_column('A:A', 25)
    worksheet.set_column('B:B', 25)
    worksheet.set_column('C:C', 25)
    cell_format = workbook.add_format()
    bold = workbook.add_format({'bold': True})
    cell_format.set_font_color('red')
    col, row = 0, 0
    for i in range(len(list_Of_Elments)):
        if (i == 5 or i == 7):
            if (i == 7):
                a = i + 4
            else:
                a = i

            if (len(arr[a]) > 1):
                worksheet.write(row, col, list_Of_Elments[i][0], bold)
                worksheet.write(row, col + 1, arr[a][0])
                row += 1
                worksheet.write(row, col, list_Of_Elments[i][1], bold)
                worksheet.write(row, col + 1, arr[a][1])
            else:  ####### possible not need all this block
                worksheet.write(row, col, list_Of_Elments[i][0], bold)
                row += 1
                worksheet.write(row, col, list_Of_Elments[i][1], bold)

        elif (i == 6):  # 6,7-10, 12-15,16-19,20-23
            worksheet.write(row, col, list_Of_Elments[i], bold)
            cnt = 1
            for j in range(6, 24):
                if (j == 11):
                    continue
                if (j == 7 or j == 12 or j == 16 or j == 20):
                    row += 1
                    cnt = 1
                    continue
                if (len(arr[j]) > 0):
                    worksheet.write(row, col + cnt, arr[j])
                cnt += 1

        elif (i == 21 or i == 22):
            worksheet.write(row, col, list_Of_Elments[i], bold)
            if (i == 22):  # 38,43,46,48,50,52
                adding = [38, 43, 46, 48, 50, 52]
            else:  # 37,42,45,47,49,51
                adding = [37, 42, 45, 47, 49, 51]
            cnnt = 1
            for j in range(len(adding)):

                if (len(arr[adding[j]]) > 1):
                    worksheet.write(row, col + cnnt, marks[j] + " " + arr[adding[j]])
                    cnnt += 1
        else:
            if (7 < i):
                a = i + 16
            else:
                a = i
            worksheet.write(row, col, list_Of_Elments[i], bold)
            if (23 == i):
                if (len(arr[a]) > 0 and (int(arr[a]) > 250 or int(arr[a]) < 30)):
                    worksheet.write(row, col + 1, arr[a], cell_format)
                else:
                    worksheet.write(row, col + 1, arr[a])
            elif (i == 24):
                if (len(arr[a]) > 0 and (int(arr[a]) > 180 or int(arr[a]) < 15)):
                    worksheet.write(row, col + 1, arr[a], cell_format)
                else:
                    worksheet.write(row, col + 1, arr[a])
            else:
                worksheet.write(row, col + 1, arr[a])
        row += 1

    # workbook.close()


def createBoxes(type):
    arr = []
    if (type == 1):
        for i in range(53):
            cell_i = BASE1_CONF.get(str(i))

            low_y = cell_i['ul'][1]
            high_y = cell_i['lr'][1]
            low_x = cell_i['ul'][0]
            high_x = cell_i['lr'][0]
            arr.append([low_x, low_y, high_x - low_x, high_y - low_y])
    else:
        for i in range(53):
            cell_i = BASE2_CONF.get(str(i))

            low_y = cell_i['ul'][1]
            high_y = cell_i['lr'][1]
            low_x = cell_i['ul'][0]
            high_x = cell_i['lr'][0]
            arr.append([low_x, low_y, high_x - low_x, high_y - low_y])

    return arr


def ocr(data_path):
    data_path = os.path.join(data_path, '1', '1')
    workbook = xlsxwriter.Workbook(data_path+'.xlsx')
    for dir_name in os.listdir(data_path):
        scan_path = os.path.join(data_path, dir_name)
        front_back = get_registered_back_and_front(scan_path)
        front = front_back["front"]["registered"]
        if not front:
            logging.error(f"Couldn't register {front_back['front']['original']}")
            return
        the_image = os.path.join(scan_path, front)
        image2excel(the_image, dir_name, workbook)
    workbook.close()
