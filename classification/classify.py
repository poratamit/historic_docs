# Imports
import cv2
import numpy as np
import numpy
from typing import List, Tuple
from functools import total_ordering
import shutil
import json
import re
import os
import logging

# Constants declaration
KNN_DISTANCE = 0.75
THRESHOLD = 10

sift = cv2.SIFT_create()
static_descriptors_path = os.path.join("classification", "static_descriptors")
with open(os.path.join(static_descriptors_path, "1_1_A.npy"), 'rb') as f:
    ver1_type1_A_des = np.load(f)
with open(os.path.join(static_descriptors_path, "1_1_B.npy"), 'rb') as f:
    ver1_type1_B_des = np.load(f)
with open(os.path.join(static_descriptors_path, "1_3_A.npy"), 'rb') as f:
    ver1_type3_A_des = np.load(f)
with open(os.path.join(static_descriptors_path, "1_3_B.npy"), 'rb') as f:
    ver1_type3_B_des = np.load(f)
with open(os.path.join(static_descriptors_path, "1_4_A.npy"), 'rb') as f:
    ver1_type4_A_des = np.load(f)
with open(os.path.join(static_descriptors_path, "1_4_B.npy"), 'rb') as f:
    ver1_type4_B_des = np.load(f)


@total_ordering
class Descriptor:
    def __init__(self, keypoint: List, desc: numpy.ndarray, src: str, version=1, subtype=1):
        self.kp = keypoint
        self.desc = desc
        self.src = src
        self.version = version
        self.subtype = subtype
        self.right = 0
        self.wrong = 0

    def is_match(self, other_descs: numpy.ndarray):
        bf_for_knn = cv2.BFMatcher()
        matches_knn = bf_for_knn.knnMatch(self.desc, other_descs, k=2)
        first_match, second_match = matches_knn[0]
        if first_match.distance < KNN_DISTANCE * second_match.distance:
            return True
        return False

    def test_desc(self, other_descs: numpy.ndarray, same_class=True):
        return self.is_match(other_descs) == same_class

    def test_and_set_desc(self, other_descs: numpy.ndarray, same_class=True):
        match = self.test_desc(other_descs, same_class)
        if match:
            self.right += 1
        else:
            self.wrong += 1

    def __hash__(self):
        return hash((self.kp, self.desc))

    def __eq__(self, other):
        return True if (self.right - self.wrong) == (other.right - other.wrong) else False

    def __gt__(self, other):
        return True if (self.right - self.wrong) > (other.right - other.wrong) else False


class Image:
    def __init__(self, path: str, version=1, subtype=1):
        self.path = path
        self.version = version
        self.subtype = subtype
        self.descriptors = numpy.ndarray([0, 128])
        self.keypoints = []
        self.des_objs = []
        self.img = None

    def read_image(self):
        self.img = cv2.imread(self.path, 0)
        return self.img

    def sift(self):
        self.keypoints, self.descriptors = sift.detectAndCompute(self.img, None)
        return self.keypoints, self.descriptors

    def create_des_objects(self):
        self.des_objs = []
        for i in range(self.descriptors.shape[0]):
            des = self.descriptors[[i], :]
            kp = self.keypoints[i]
            self.des_objs.append(Descriptor(kp, des, self.path))

    @staticmethod
    def sift_on_files(paths: List[str]) -> List[np.ndarray]:
        des_list = []
        for path in paths:
            img = cv2.imread(path, cv2.COLOR_BGR2BGRA)
            _, des = sift.detectAndCompute(img, None)
            logging.debug(f"Ran SIFT on {path}")
            des_list.append(des)
        return des_list

    def test_descriptors(self, others: List[np.ndarray], ours=None, same_class=True):
        if not ours:
            ours = self.des_objs
        for descriptor in ours:
            for other in others:
                descriptor.test_and_set_desc(other, same_class=same_class)


class Dir:
    def __init__(self, source: str, dest: str):
        self.source = source
        self.dest = dest
        self.files = []
        self.classified = set()
        self.results = {1: {
            1: [],
            3: [],
            4: []
        },
            'unclassified': []}
        self.matching_results = {}

    def get_files(self) -> List:
        """
        Return a list of file paths, sorted by the page number in the PDF
        :return:
        """
        pics = [filename for filename in os.listdir(self.source) if filename.endswith(".jpg")]
        self.files = [f"{self.source}/{filename}" for filename in
                      sorted(pics, key=lambda x: int(re.search(r".*?(\d+)\.jpg", x).group(1)))]
        self.unclassified = set(self.files)
        return self.files

    @staticmethod
    def test_descriptors(path: str) -> List[Tuple[str, int]]:
        """
        For each type, count the number of descriptors which match the input file's descriptors
        :param path: path to the file which is about to be classified
        :return:
        """
        des_count = {
            "ver_1type1_A": [ver1_type1_A_des, 0],
            "ver_1type1_B": [ver1_type1_B_des, 0],
            "ver_1type3_A": [ver1_type3_A_des, 0],
            "ver_1type3_B": [ver1_type3_B_des, 0],
            "ver_1type4_A": [ver1_type4_A_des, 0],
            "ver_1type4_B": [ver1_type4_B_des, 0],
        }

        file_descs = Image.sift_on_files([path])[0]
        for des_ver in des_count:
            for des in des_count[des_ver][0]:
                # TODO: change Descriptor
                des_obj = Descriptor([], des, "1_1.npy")
                if des_obj.is_match(file_descs):
                    des_count[des_ver][1] += 1
        return sorted([(des_ver, des_count[des_ver][1]) for des_ver in des_count], key=lambda x: x[1], reverse=True)

    def classify_dir(self):
        """
        Returns for each file the number of descriptors matches for each of the types
        :return:
        """
        for path in self.files:
            self.matching_results[path] = Dir.test_descriptors(path)
        return self.matching_results

    def get_results(self):
        """
        Classify the files based on the descriptors matching
        :return:
        """
        for i in range(len(self.files)):
            cur_file = self.files[i]
            if i == len(self.files) - 1 or cur_file in self.classified:
                continue
            next_file = self.files[i + 1]
            cur_result = self.matching_results[cur_file][0]
            next_result = self.matching_results[next_file][0]

            # The current file and the next file are of the same type, where the first one is the front and the next
            # one is the back
            if cur_result[0][:-1] == next_result[0][:-1] and cur_result[0][-1] == "A" and next_result[0][-1] == "B" and \
                    cur_result[1] >= THRESHOLD and next_result[1] >= THRESHOLD:
                type_id = int(cur_result[0][-3])
                self.results[1][type_id].append((cur_file, next_file))
                self.classified.add(cur_file)
                self.classified.add(next_file)
                logging.debug(f"Classified {cur_file} and {next_file} as type {type_id}")
            else:
                logging.debug(f"Classified {cur_file} as others")
                self.results['unclassified'].append(cur_file)
        return self.results

    def save_results(self):
        type_dirs = [os.path.join(self.dest, '1', str(i)) for i in (1, 3, 4)]
        for dir_path in type_dirs:
            os.makedirs(dir_path, exist_ok=True)
        ver1 = self.results.get(1)
        for type_id in ver1:
            type_path = os.path.join(self.dest, '1', str(type_id))
            pairs = ver1.get(type_id)
            for i, pair in enumerate(pairs):
                pair_path = os.path.join(type_path, str(i))
                os.makedirs(pair_path, exist_ok=True)
                for path in pair:
                    shutil.copyfile(path, os.path.join(pair_path, os.path.basename(path)))
                    log = self.matching_results.get(path)
                    with open(os.path.join(pair_path, os.path.basename(path)) + ".json", "w") as f:
                        json.dump(log, f)
        others_path = os.path.join(self.dest, 'others')
        os.makedirs(others_path, exist_ok=True)
        for other in self.results.get("unclassified"):
            shutil.copyfile(other, os.path.join(others_path, os.path.basename(other)))
            log = self.matching_results.get(other)
            with open(os.path.join(others_path, os.path.basename(other)) + ".json", "w") as f:
                json.dump(log, f)


def classify(src_dir: str, dst_dir: str):
    dir_obj = Dir(src_dir, dst_dir)
    dir_obj.get_files()
    classified = dir_obj.classify_dir()
    dir_obj.get_results()
    dir_obj.save_results()