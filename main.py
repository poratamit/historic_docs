import cv2
from classification.classify import classify
from racial_stats.racial_stats import get_stats
import logging
from registration.register import registerFile
import os
from utils import get_registered_back_and_front
import sys
from cropping.crop import crop
#logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s %(levelname)s :%(filename)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

dir_path = r"C:\Users\dell\Downloads\R 9361 IV_EWZ 56_K17_classified-20210913T101130Z-001\R 9361 IV_EWZ 56_K17_classified"
#classify(r"C:\Users\dell\OneDrive\Desktop\studies\3rd year\2nd Semester\ML_Workshop\R 9361 IV (EWZ 56) G34_part_1", "R 9361 IV (EWZ 56) G34_part_1-classified")
# #get_stats(r"C:\Users\dell\Downloads\E 9361 IV (EWZ 56)_Nr. A1_all in 1_registerd-20210911T170939Z-001\E 9361 IV (EWZ 56)_Nr. A1_all in 1_registerd")
# #registerFile(r"C:\Users\dell\Downloads\R 9361 IV_EWZ 56_K17_classified-20210912T103004Z-001\R 9361 IV_EWZ 56_K17_classified")
# #registerFile(dir_path)
#get_stats(dir_path)
#get_registered_back_and_front(r"C:\Users\dell\Downloads\R 9361 IV_EWZ 56_K17_classified-20210913T101130Z-001\R 9361 IV_EWZ 56_K17_classified\1\1\24")
crop(dir_path)