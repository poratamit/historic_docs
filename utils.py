import os
import re
import logging


def get_registered_back_and_front(scan_path: str):
    result = {"front": {"original": "", "registered": ""}, "back": {"original": "", "registered": ""}}
    pics = []
    for file_name in os.listdir(scan_path):
        if file_name.endswith(".jpg"):
            if re.match(r".*[\_|\-]\d+\-Base\d+\.jpg", file_name) or re.match(r".*[\_|\-]\d+\-Face.jpg", file_name) or re.match(r".*[\_|\-]\d+\-Dumped\d*.jpg", file_name):
                continue
            pics.append(file_name)
    if len(pics) < 2:
        return result

    match_0 = re.match(r".*[\_|\-](\d+).jpg", pics[0])
    match_1 = re.match(r".*[\_|\-](\d+).jpg", pics[1])
    if not match_0 or not match_1:
        logging.error(f"File names not in correct format {pics[0]} {pics[1]}")
        return result
    pic_num_0 = int(match_0.group(1))
    pic_num_1 = int(match_1.group(1))
    if pic_num_0 > pic_num_1:
        result["back"]["original"] = pics[0]
        result["front"]["original"] = pics[1]
    else:
        result["back"]["original"] = pics[1]
        result["front"]["original"] = pics[0]

    for file_name in os.listdir(scan_path):
        if file_name.endswith(".jpg"):
            if result["back"]["original"].strip(".jpg") in file_name and "Base" in file_name:
                result["back"]["registered"] = file_name
            elif result["front"]["original"].strip(".jpg") in file_name and "Base" in file_name:
                result["front"]["registered"] = file_name
    return result
    # Get the back img
    # match_0 = re.match(r".*[\_|\-](\d+)\-(Base\d+)\.jpg", pics[0])
    # pic_num_0, base_0 = int(match_0.group(1)), match_0.group(2)
    # match_1 = re.match(r".*[\_|\-](\d+)\-(Base\d+)\.jpg", pics[1])
    # pic_num_1, base_1 = int(match_1.group(1)), match_1.group(2)
    # img = pics[0] if pic_num_0 > pic_num_1 else pics[1]