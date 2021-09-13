import cv2
import numpy as np
import os
import re

# Types with faces being used
faceTypes = ["type1front"]

# Total types being used
types = ["type1back", "type1front"]

# Creating SIFT and matcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Load the face recognition cascade
face_cascade = cv2.CascadeClassifier(os.path.join('registration, haarcascade_frontalface_default.xml'))


# Return all the names of the images in a classification file
# Input: File name
# Output: a list in that order [[front1,back1],[front2,back2]...]
def get_names(file):
    name_list = []
    #file = file + "/1"
    file = os.path.join(file, '1')
    for k in os.listdir(file):
        if k != "1":
            continue
        for dir_name in os.listdir(os.path.join(file, k)):
            pics = []
            for file_name in os.listdir(os.path.join(file, k, dir_name)):
                file_path = "".join([os.path.join(dir_name, file_name)])
                if file_path.endswith(".jpg"):
                    pics.append(file_path)
            try:
                pic_num_0 = int(re.match(".*\-(\d+)\.jpg", pics[0]).group(1))
                pic_num_1 = int(re.match(".*\-(\d+)\.jpg", pics[1]).group(1))
            except:
                try:
                    pic_num_0 = int(re.match(".*\_(\d+)\.jpg", pics[0]).group(1))
                    pic_num_1 = int(re.match(".*\_(\d+)\.jpg", pics[1]).group(1))
                except:
                    continue
            # print(pics[0] if pic_num_0 < pic_num_1 else pics[1])
            base_path = os.path.join(file, str(k))
            if (pic_num_0 < pic_num_1):
                names = []
                names.append(os.path.join(base_path, pics[0]))
                names.append(os.path.join(base_path, pics[1]))
                name_list.append(names)
            else:
                names = []
                names.append(os.path.join(base_path, pics[1]))
                names.append(os.path.join(base_path, pics[0]))
                name_list.append(names)
    return name_list


# Create keypoints and descriptors of the bases using SIFT for a single type
def prepBasesSIFT(imType):
    basesKeypointsSIFT, basesDescriptorsSIFT = [], []
    bases = []
    # Iterate on all the bases in a single type file
    for i in range(100):
        fileName = os.path.join("registration", "bases", imType, f'base{i+1}.jpg')
        #fileName = "".join(["bases/", imType, "/base", str(i + 1), ".jpg"])
        base = cv2.imread(fileName, 0)
        if base is None:
            continue
        bases.append(fileName)
        keypoints, descriptors = sift.detectAndCompute(base, None)
        basesKeypointsSIFT.append(keypoints)
        basesDescriptorsSIFT.append(descriptors)
    return basesKeypointsSIFT, basesDescriptorsSIFT, bases


# Create keypoints and descriptors of all the bases (all types)
def prepBases():
    bases = {}
    basesKey = {}
    basesDes = {}
    for imType in types:
        basesKeypointsSIFT, basesDescriptorsSIFT, bases1 = prepBasesSIFT(imType)
        bases[imType] = bases1
        basesKey[imType] = basesKeypointsSIFT
        basesDes[imType] = basesDescriptorsSIFT
    return bases, basesKey, basesDes


# Allign using sift 
def align(im, base, keypoints1, descriptors1, keypoints2, descriptors2):

    # Match features using knn
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Choose good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            # Extract location of good matches
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)
    for i, match in enumerate(good):
        points1[i, :] = keypoints1[match[0].queryIdx].pt
        points2[i, :] = keypoints2[match[0].trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width = base.shape
    imReg = cv2.warpPerspective(im, h, (width, height))
    return imReg, h


# Get alligment score using SIFT on a registered image
# Output: determinant score
def testAlignment(keypoints1, descriptors1, keypoints2, descriptors2):
    # Match features using knn
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Choose good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            # Extract location of good matches
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)
    for i, match in enumerate(good):
        points1[i, :] = keypoints1[match[0].queryIdx].pt
        points2[i, :] = keypoints2[match[0].trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use Determinante
    d = np.linalg.det(h)
    d = abs(1 - d)
    return d


# Detect face if the type of the image is in the faceTypes list
def faceDetection(im, imName):
    # Detect faces
    faces = face_cascade.detectMultiScale(im, 1.1, 4)
    # Get original image shape
    height, width = im.shape
    # Cut Face
    if len(faces) == 0:
        return
    for (x, y, w, h) in faces:
        cropImage = im[np.maximum(y - 300, 0):np.minimum(y + h + 300, height),
                    np.maximum(x - 300, 0):np.minimum(x + w + 300, width)]
        break
    # Display the output
    fileName = "".join([imName, "-Face.jpg"])
    cv2.imwrite(fileName, cropImage)
    print("Face: " + fileName)


"""
# Register single image by a given base
# Input:  path, type, bases list, bases keypoints list, bases descriptor list, treshhold of the determinant score
# The function save the image after allignment with the corresponding name : base1,base2,dumped
"""


# You must have the files "bases", "haarcascade_frontalface_default.xml", and the file with the images in the same file as the code
def registerSingle(imName, imType, bases, basesKey, basesDes, treshhold):
    bases = bases[imType]
    basesKey = basesKey[imType]
    basesDes = basesDes[imType]
    im = cv2.imread(imName, 0)
    # imName = imName.split(".")
    # if len(imName) > 2:
    #     imName = imName[0] + "." + imName[1]
    # else:
    #     imName = imName[0]
    imName = imName.strip(".jpg")
    if im is None:
        print("Error: open im")
        exit()
    # Detect keypoints in the image
    keypoints, descriptors = sift.detectAndCompute(im, None)
    bestIm, bestD, bestBase = 0, 100, -1
    if imType in faceTypes:
        # Detect face
        faceDetection(im, imName)
    # Iterate on all the bases and get for each base determinant score
    for i in range(len(bases)):
        try:
            base = cv2.imread(bases[i], 0)
            imReg, a = align(im, base, keypoints, descriptors, basesKey[i], basesDes[i])
            regKeypoints, regDescriptors = sift.detectAndCompute(imReg, None)
            d = testAlignment(regKeypoints, regDescriptors, basesKey[i], basesDes[i])
        except:
            continue
        # Choose the best base
        print("base" + str(i + 1) + ": " + str(d))
        if (d < bestD):
            bestD = d
            bestIm = imReg
            bestBase = i + 1
    # Compare to treshhold and decide
    if bestD < treshhold:
        outFilename = "".join([imName, "-Base", str(bestBase), ".jpg"])
        cv2.imwrite(outFilename, bestIm)
        print("Out: " + outFilename)
        return bestBase
    else:
        outFilename = "".join([imName, "-Dumped", str(bestBase), ".jpg"])
        cv2.imwrite(outFilename, bestIm)
        print("Dumped: " + outFilename)
        return -1


# Register a whole file one image by another.
# Can take few hours for alot of images
def registerFile(file):
    bases, basesKey, basesDes = prepBases()
    frontbase1, frontbase2, backbase1, backbase2, backbase3, frontdump, backdump = 0, 0, 0, 0, 0, 0, 0
    print("Bases have been computed")
    index = 0
    names = get_names(file)
    for name in names:
        print()
        index += 1
        print(index)
        print("File: " + name[0])
        bestBase = registerSingle(name[0], "type1front", bases, basesKey, basesDes, 0.15)
        if (bestBase == 1):
            frontbase1 += 1
        elif (bestBase == 2):
            frontbase2 += 1
        else:
            frontdump += 1
        try:
            print()
            index += 1
            print(index)
            print("File: " + name[1])
            bestBase = registerSingle(name[1], "type1back", bases, basesKey, basesDes, 0.05)
            if (bestBase == 1):
                backbase1 += 1
            elif (bestBase == 2):
                backbase2 += 1
            elif (bestBase == 3):
                backbase3 += 1
            else:
                backdump += 1
        except:
            continue
    # Statistics
    print("index: " + str(index))
    print("frontbase1:  " + str(frontbase1))
    print("frontbase2:  " + str(frontbase2))
    print("frontdump:  " + str(frontdump))
    print("backbase1:  " + str(backbase1))
    print("backbase2:  " + str(backbase2))
    print("backbase3:  " + str(backbase3))
    print("backdump:  " + str(backdump))
