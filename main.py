import cv2
from classification.classify import classify
from racial_stats.racial_stats import get_stats
import logging
from registration.register import registerFile
logging.basicConfig(filename="myapp.log", level=logging.INFO)


#classify(r"C:\Users\dell\OneDrive\Desktop\studies\3rd year\2nd Semester\ML_Workshop\R 9361 IV (EWZ 56) G34_part_1", "R 9361 IV (EWZ 56) G34_part_1-classified")
get_stats(r"C:\Users\dell\Downloads\E 9361 IV (EWZ 56)_Nr. A1_all in 1_registerd-20210911T170939Z-001\E 9361 IV (EWZ 56)_Nr. A1_all in 1_registerd")
#registerFile(r"C:\Users\dell\Downloads\R 9361 IV_EWZ 56_K17_classified-20210912T103004Z-001\R 9361 IV_EWZ 56_K17_classified")

