# Imports
import numpy as np
import cv2
from typing import Tuple

# Declaring constants
WHITE_PIXEL = 255
LINE_MIN_WIDTH = 15


class Rec:
    def __init__(self, rec: np.ndarray, i, j):
        self.rec = rec
        self.i = i
        self.j = j
        self.max_y = rec.shape[0]
        self.max_x = rec.shape[1]

    def show(self):
        cv2_imshow(self.rec)

    def get_max_white(self, width: int, jump: int, min_y=0, max_y=None, min_x=0, max_x=None) -> Tuple[int, float]:
        max_y = self.max_y if not max_y else max_y
        max_x = self.max_x if not max_x else max_x
        max_white_pixels = 0
        x1 = 0
        x2 = width
        place = 0
        while x1 < max_x:
            cropped = self.rec[min_y:max_y, x1:x2]
            n_white_pixels = np.sum(cropped >= 225)
            if n_white_pixels > max_white_pixels:
                # place = (x1 + x2) / max_x
                place = x1 / max_x
                max_white_pixels = n_white_pixels
            x1 += jump
            x2 += jump
        return (max_white_pixels, place)


class BinaryRec(Rec):
    def __init__(self, rec: np.ndarray, i, j):
        Rec.__init__(self, rec, i, j)


class VerticalCleanedRec(Rec):
    def __init__(self, rec: np.ndarray, i, j):
        Rec.__init__(self, rec, i, j)


class CornersImage(Rec):
    def __init__(self, rec: np.ndarray, i, j):
        Rec.__init__(self, rec, i, j)


class CleanRec(Rec):
    def __init__(self, rec: np.ndarray, i, j):
        Rec.__init__(self, rec, i, j)

    def get_corners(self, corners_num: int) -> Tuple[np.ndarray, CornersImage]:
        marked_image = self.rec.copy()
        corners = cv2.goodFeaturesToTrack(marked_image, corners_num, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(marked_image, (x, y), 3, 255, -1)
        return corners, CornersImage(marked_image, self.i, self.j)

    def get_verticals(self) -> VerticalCleanedRec:
        kernal_v = np.ones((LINE_MIN_WIDTH + 18, 1), np.uint8)
        img_bin_v = cv2.morphologyEx(~self.rec, cv2.MORPH_OPEN, kernal_v)
        final_kernel = np.ones((10, 4), np.uint8)
        img_bin_v = cv2.dilate(img_bin_v, final_kernel, iterations=1)
        return VerticalCleanedRec(img_bin_v, self.i, self.j)


class GrayRec(Rec):
    def __init__(self, rec: np.ndarray, i, j):
        Rec.__init__(self, rec, i, j)

    def remove_noise_from_rec(self) -> CleanRec:
        # th1,img_bin = cv2.threshold(self.rec,150,225,cv2.THRESH_BINARY)
        th1, img_bin = cv2.threshold(self.rec, 150, 225, cv2.THRESH_BINARY)
        img_bin = ~img_bin

        kernal_h = np.ones((1, LINE_MIN_WIDTH + 40), np.uint8)
        # kernal_v = np.ones((LINE_MIN_WIDTH+15,1), np.uint8)
        kernal_v = np.ones((LINE_MIN_WIDTH + 10, 1), np.uint8)

        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
        img_bin_final = img_bin_h | img_bin_v

        final_kernel = np.ones((3, 4), np.uint8)
        # img_bin_final = cv2.erode(img_bin_final, final_kernel, iterations=1)
        # img_bin_final = cv2.dilate(img_bin_final,final_kernel,iterations=1)
        return CleanRec(~img_bin_final, self.i, self.j)