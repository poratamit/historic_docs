# Imports
import cv2
import numpy as np
from typing import Tuple, List
from .rectangles import GrayRec, Rec, BinaryRec
import json


class Image:
    def __init__(self, path_to_img: str):
        self.img_path = path_to_img
        self.img = None
        self.gray_scale = None
        self.kp = None
        self.des = None
        self.read_image()
        self.run_sift()

    def read_image(self):
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise FileNotFoundError(f"Couldn't read {self.img_path}")
        self.gray_scale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def run_sift(self) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        sift = cv2.SIFT_create()
        self.kp, self.des = sift.detectAndCompute(self.gray_scale, None)
        return self.kp, self.des

    def get_rectangles_by_conf(self, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                               offset_down=0) -> dict:
        return {(int(i), int(j)): self.get_gray_rectangle(i, j, base_image, offset_left, offset_right, offset_up,
                                                          offset_down) for i in base_image.conf for j in
                base_image.conf[i]}

    def get_binary_rectangles_by_conf(self, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                                      offset_down=0) -> dict:
        return {(int(i), int(j)): self.get_binary_rectangle(i, j, base_image, offset_left, offset_right, offset_up,
                                                            offset_down) for i in base_image.conf for j in
                base_image.conf[i]}

    def get_rectangle(self, i: str, j: str, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                      offset_down=0) -> np.ndarray:
        row_conf = base_image.conf.get(i)
        if not row_conf:
            raise ValueError(f"Invalid row number {i}")
        rec_conf = row_conf.get(j)
        if not rec_conf:
            raise ValueError(f"Invalid column number {j}")
        low_y = rec_conf['ul'][1]
        high_y = rec_conf['lr'][1]
        low_x = rec_conf['ul'][0]
        high_x = rec_conf['lr'][0]
        return self.img[low_y - offset_down: high_y + offset_up, low_x - offset_left:high_x + offset_right]

    def get_gray_rectangle(self, i: str, j: str, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                           offset_down=0) -> GrayRec:
        return GrayRec(self.get_rectangle(i, j, base_image, offset_left, offset_right, offset_up, offset_down), i, j)

    def get_col(self, i: str, j: str, base_image: 'BaseImage', offset_left=5, offset_right=5, offset_up=0,
                offset_down=0) -> np.ndarray:
        row_conf = base_image.conf.get(i)
        if not row_conf:
            raise ValueError(f"Invalid row number {i}")
        rec_conf = row_conf.get(j)
        if not rec_conf:
            raise ValueError(f"Invalid column number {j}")
        low_y = rec_conf['ul'][1]
        high_y = rec_conf['lr'][1]
        low_x = rec_conf['ul'][0]
        return self.img[low_y - offset_down: high_y + offset_up, low_x - offset_left:low_x + offset_right]

    def get_gray_col(self, i: str, j: str, base_image: 'BaseImage', offset_left=5, offset_right=5, offset_up=0,
                     offset_down=0) -> GrayRec:
        return GrayRec(self.get_col(i, j, base_image, offset_left, offset_right, offset_up, offset_down), i, j)

    def get_binary_col(self, i: str, j: str, base_image: 'BaseImage', offset_left=5, offset_right=5, offset_up=0,
                       offset_down=0) -> BinaryRec:
        cropped_col = self.get_col(i, j, base_image, offset_left, offset_right, offset_up, offset_down)
        th1, img_bin = cv2.threshold(cropped_col, 170, 225, cv2.THRESH_BINARY)
        return BinaryRec(img_bin, i, j)

    def get_cols_by_conf(self, base_image: 'BaseImage', offset_left=5, offset_right=5, offset_up=0, offset_down=0) -> \
            dict:
        return {(int(i), int(j)): self.get_gray_col(i, j, base_image, offset_left, offset_right, offset_up, offset_down)
                for i in base_image.conf for j in base_image.conf[i]}

    def get_binary_col_by_conf(self, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                               offset_down=0) -> dict:
        return {
            (int(i), int(j)): self.get_binary_col(i, j, base_image, offset_left, offset_right, offset_up, offset_down)
            for i in base_image.conf for j in base_image.conf[i]}

    def get_binary_rectangle(self, i: str, j: str, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                             offset_down=0) -> BinaryRec:
        cropped_rec = self.get_rectangle(i, j, base_image, offset_left, offset_right, offset_up, offset_down)
        th1, img_bin = cv2.threshold(cropped_rec, 170, 225, cv2.THRESH_BINARY)
        return BinaryRec(img_bin, i, j)

    def get_rectangles_row(self, i: str, base_image: 'BaseImage', offset_left=0, offset_right=0, offset_up=0,
                           offset_down=0) -> dict:
        return {
            (int(i), j): self.get_rectangle(i, str(j), base_image, offset_left, offset_right, offset_up, offset_down)
            for j in range(1, 6)}


class BaseImage(Image):
    def __init__(self, path_to_img: str, conf: dict):
        Image.__init__(self, path_to_img)
        self.conf = conf


class InputImage(Image):
    def __init__(self, path_to_img: str):
        self.aligned = None
        Image.__init__(self, path_to_img)

    def align(self, base_image: BaseImage) -> np.ndarray:

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des, base_image.des, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append([m])

        # Extract location of good matches
        points1 = np.zeros((len(good), 2), dtype=np.float32)
        points2 = np.zeros((len(good), 2), dtype=np.float32)

        for i, match in enumerate(good):
            points1[i, :] = self.kp[match[0].queryIdx].pt
            points2[i, :] = base_image.kp[match[0].trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width = base_image.gray_scale.shape
        self.aligned = cv2.warpPerspective(self.gray_scale, h, (width, height))

        return self.aligned

    @staticmethod
    def get_vertical_rectangles(recs: dict):
        return {(int(rec.i), int(rec.j)): rec.remove_noise_from_rec().get_verticals() for rec in recs.values()}

    def get_binary_subtract_by_conf(self, base_image: BaseImage, offset_left=0, offset_right=0, offset_up=0,
                                    offset_down=0) -> dict:
        binary_recs = self.get_binary_rectangles_by_conf(base_image, offset_left, offset_right, offset_up, offset_down)
        base_binary_recs = base_image.get_binary_rectangles_by_conf(base_image, offset_left, offset_right, offset_up,
                                                                    offset_down)
        return {(i, j): BinaryRec(cv2.subtract(base_binary_recs[i, j].rec, binary_recs[i, j].rec), i, j) for i, j in
                binary_recs}

    def get_binary_subtract_col_by_conf(self, base_image: BaseImage, offset_left=0, offset_right=0, offset_up=0,
                                        offset_down=0) -> dict:
        binary_cols = self.get_binary_col_by_conf(base_image, offset_left, offset_right, offset_up, offset_down)
        base_binary_cols = base_image.get_binary_col_by_conf(base_image, offset_left, offset_right, offset_up,
                                                             offset_down)
        return {(i, j): BinaryRec(cv2.subtract(base_binary_cols[i, j].rec, binary_cols[i, j].rec), i, j) for i, j in
                binary_cols}

    @staticmethod
    def get_binary_rec(rec: Rec):
        th1, img_bin = cv2.threshold(rec.rec, 150, 225, cv2.THRESH_BINARY)
        return img_bin

    @staticmethod
    def get_best_in_row(recs, sub_recs, row: int, width: int, jump: int):
        results = [(recs[row, i].get_max_white(width, jump), i) for i in range(1, 6)]
        results.sort(key=lambda x: x[0][0], reverse=True)
        if results[0][0][0] > 1.35 * results[1][0][0]:
            # print("In vertical ", results[0][0][0], results[1][0][0])
            return (results[0][1], results[0][0][1])
        else:
            return InputImage.get_best_subtr_in_row(sub_recs, row, width, jump)

    @staticmethod
    def get_best_in_row_with_col(recs, sub_recs, sub_cols, row: int, width: int, jump: int, col_width: int,
                                 col_jump: int):
        col, place = InputImage.get_best_subtr_col_in_row(sub_cols, row, col_width, col_jump)
        if (col, place) != (-1, -1):
            return col, place
        return InputImage.get_best_in_row(recs, sub_recs, row, width, jump)

    @staticmethod
    def get_best_column(cols, sub_cols, row: int, width: int, jump: int):
        results = [(cols[row, i].get_max_white(width, jump), i) for i in range(1, 6)]
        results.sort(key=lambda x: x[0][0], reverse=True)
        if results[0][0][0] > 1.5 * results[1][0][0]:
            return (results[0][1], 0.0)
        else:
            return InputImage.get_best_subtr_col_in_row(sub_cols, row, width, jump)

    @staticmethod
    def get_best_subtr_col_in_row(sub_cols, row: int, width: int, jump: int):
        results = [(sub_cols[row, i].get_max_white(width, jump), i) for i in range(1, 6)]
        results.sort(key=lambda x: x[0][0], reverse=True)
        # print("Col status", results[0][0][0], results[1][0][0])
        if results[0][0][0] > 3.75 * results[1][0][0] and results[0][0][0] > 200:
            # print("In Col ", results[0][0][0], results[1][0][0])
            return (results[0][1], 0.0)
        else:
            return (-1, -1)

    @staticmethod
    def get_best_subtr_in_row(recs, row: int, width: int, jump: int):
        results = [(recs[row, i].get_max_white(width, jump), i) for i in range(1, 6)]
        results.sort(key=lambda x: x[0][0], reverse=True)
        if results[0][0][0] > 1.75 * results[1][0][0]:
            # print("In subtr ", results[0][0][0], results[1][0][0])
            return (results[0][1], results[0][0][1])
        else:
            return (-1, -1)