from .conf import FRONT, BACK
import re
import os
import logging
from utils import get_registered_back_and_front
import cv2

mapping = {"front": FRONT, "back": BACK}


def crop(dir_path: str):
    dir_path = os.path.join(dir_path, '1', '1')
    for dir_name in os.listdir(dir_path):
        scan_path = os.path.join(dir_path, dir_name)
        front_back = get_registered_back_and_front(scan_path)
        front = front_back["front"]["registered"]
        if not front:
            logging.error(f"Couldn't register {front_back['front']['original']}")
        else:
            crop_side(scan_path, "front", front)
        back = front_back["back"]["registered"]
        if not back:
            logging.error(f"Couldn't register {front_back['back']['original']}")
        else:
            crop_side(scan_path, "back", back)


def crop_side(scan_path, side_name: str, file_name: str):
    conf = mapping.get(side_name)
    if not conf:
        logging.error("Invalid side, only front/back")
        return
    match_0 = re.match(r".*[\_|\-]\d+\-(Base\d+)\.jpg", file_name)
    base_name = match_0.group(1)
    base_conf = conf.get(base_name)
    if not base_conf:
        logging.error(f"Couldn't extract base of {file_name}")
        return
    file_path = os.path.join(scan_path, file_name)
    img = cv2.imread(file_path)
    for snip in base_conf:
        coordinates = base_conf[snip]
        low_y = coordinates['ul'][1]
        high_y = coordinates['lr'][1]
        low_x = coordinates['ul'][0]
        high_x = coordinates['lr'][0]
        cropped = img[low_y: high_y, low_x:high_x]
        cropped_path = os.path.join(scan_path, f"{snip}.jpg")
        try:
            cv2.imwrite(cropped_path, cropped)
        except:
            logging.error(f"Couldn't save {cropped_path}")
        else:
            logging.debug(f"Saved {cropped_path}")



