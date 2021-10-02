from classification.classify import classify
from racial_stats.racial_stats import get_stats
import logging
from registration.register import registerFile
from utils import get_registered_back_and_front
import argparse
from cropping.crop import crop
import os
from ocr.ocr import ocr

def main():
    # logger = logging.getLogger()
    # logger.setLevel(level=logging.DEBUG)
    # fh = logging.StreamHandler()
    # fh_formatter = logging.Formatter('%(asctime)s %(levelname)s :%(filename)s - %(message)s')
    # fh.setFormatter(fh_formatter)
    # logger.addHandler(fh)
    #
    # parser = argparse.ArgumentParser(description='Extract data from historic documents')
    # parser.add_argument('-s', '--src', help="Source dir, for example: R 9361 IV_EWZ 56_K17")
    # parser.add_argument('-d', '--dst', help="Destination dir for classification, face images, "
    #                                         "cropping and data extraction. Only needed if -n is not specified")
    # parser.add_argument("-n", "--no-register", help="Use this flag if the source directory is already registered", action="store_true")
    # args = parser.parse_args()
    # src_dir = args.src
    # dst_dir = args.dst
    # if not args.no_register:
    #     classify(src_dir, dst_dir)
    #     registerFile(dst_dir)
    # else:
    #     dst_dir = src_dir
    # get_stats(dst_dir, os.path.join(dst_dir, os.path.basename(dst_dir) + ".xlsx"))
    # crop(dst_dir)

    ocr(r"C:\Users\dell\OneDrive\Desktop\studies\3rd year\2nd Semester\ML_Workshop\Registration-20210913T171125Z-001\Registration\R 9361 IV (EWZ 56) G34_part_1_registerd - Copy")

if __name__ == "__main__":
    main()