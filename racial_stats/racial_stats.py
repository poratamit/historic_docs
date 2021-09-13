# Imports
from .images import BaseImage, InputImage
from .conf.conf import BASE1_CONF, BASE2_CONF, BASE3_CONF
import xlsxwriter
import os
from utils import get_registered_back_and_front
from collections import OrderedDict
import logging
import re

# Declaring constants
COLUMNS = {1: '1 - Körperhöhe', 2: '2 - Wuchsform', 3: '3 - Haltung', 4: '4 - Beinlänge rel.', 5: '5 - Kopfform', 6: '6 - Hinterhaupt', 7: '7 - Gesichtsform', 8: '8 - Nasenrüeken', 9: '9 - Nasenhöhe', 10: '10 - Nasenbreite', 11: '11 - Backenknochen', 12: '12 - Augenlage', 13: '13 - Lidspalte', 14: '14 - Augenfalten-bildung', 15: '15 - Lippen', 16: '16 - Kinn', 17: '17 - Haarform', 18: '18 - Körperbehaarg.', 19: '19 - Haarfarbe', 20: '20 - Augenfarbe', 21: '21 - Hautfarbe'}

type1back_path = os.path.join("registration", "bases", "type1back")
base1 = BaseImage(os.path.join(type1back_path, "base1.jpg"), BASE1_CONF)
base2 = BaseImage(os.path.join(type1back_path, "base2.jpg"), BASE2_CONF)
base3 = BaseImage(os.path.join(type1back_path, "base3.jpg"), BASE3_CONF)
bases = {
    "Base1": base1,
    "Base2": base2,
    "Base3": base3
}


def get_stats(dir_path: str, file_path=None):
    file_path = file_path if file_path else f'{os.path.basename(dir_path)}.xlsx'
    results = {}
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    worksheet.write(0, 0, "File Name", bold)
    for i in range(1, 22):
        worksheet.write(0, i, COLUMNS[i], bold)

    cnt = 1
    dir_path = os.path.join(dir_path, '1', '1')
    dir_names = sorted([int(i) for i in os.listdir(dir_path)])
    for dir_name in dir_names:
        dir_name = str(dir_name)
        scan_path = os.path.join(dir_path, dir_name)
        front_back = get_registered_back_and_front(scan_path)
        img = front_back["back"]["registered"]
        if not img:
            logging.error(f"Couldn't register {front_back['back']['original']}")
            continue
        match_0 = re.match(r".*[\_|\-]\d+\-(Base\d+)\.jpg", img)
        base_name = match_0.group(1)
        base = bases.get(base_name)
        #base = bases.get(base_0 if pic_num_0 > pic_num_1 else base_1)
        file_path = os.path.join(scan_path, img)
        logging.debug(f"Analyzing {img}")
        input_image = InputImage(file_path)
        sub_cols = input_image.get_binary_subtract_col_by_conf(base, offset_left=7, offset_right=7)
        subtr = input_image.get_binary_subtract_by_conf(base, offset_left=-7, offset_right=-7, offset_up=0,
                                                        offset_down=0)
        rectangles = input_image.get_rectangles_by_conf(base, offset_left=-7, offset_right=-7, offset_up=0,
                                                        offset_down=0)
        verticals = InputImage.get_vertical_rectangles(rectangles)
        results[img] = OrderedDict()
        for i in range(1, 22):
            col, place = InputImage.get_best_in_row_with_col(verticals, subtr, sub_cols, i, 10, 5, 10, 1)
            row_result = float(f"{col}.{str(place).split('.')[1]}") if (col, place) != (-1, -1) else -1
            results[img][COLUMNS[i]] = row_result
            worksheet.write(cnt, i, row_result)
        worksheet.write(cnt, 0, int(dir_name))
        cnt += 1
    workbook.close()