import os
from os import path
import glob
import cv2
import numpy as np
import imutils
import random

total_num = 600
image_folder = "/home/indra/Documents/XRVision/imgcls/dataset/"
faulty = image_folder + "faulty/"
if not path.exists(faulty):
    os.mkdir(faulty)
not_faulty = image_folder + "not_faulty/"
if not path.exists(not_faulty):
    os.mkdir(not_faulty)
angles = [90, 180, 270]

# rename current images in faulty folder (DONE)
j = 1
for f1 in os.listdir(faulty):
    image = cv2.imread(f1)
    print(f1)
    cv2.imwrite(faulty + "chips (" + str(j) + ").jpg", image)
    os.remove(faulty + f1)
    j = j + 1

# move files into non_faulty folder
for f1 in os.listdir(image_folder):
    if f1.endswith('.jpg'):
        os.rename(image_folder + f1, not_faulty + f1)

# rotate not_faulty images to create faulty images
for i in range(11, total_num+1):
    print(i)
    image = cv2.imread(not_faulty + "chips (" + str(i) + ").jpg")
    rotated = imutils.rotate(image, angles[random.randint(0,2)])
    cv2.imwrite(faulty + "chips (" + str(i) + ").jpg", rotated)

# create train, val, test folders
os.mkdir(image_folder + "train")
train_faulty = image_folder + "train/faulty/" 
os.mkdir(train_faulty)
train_not_faulty = image_folder + "train/not_faulty/"
os.mkdir(train_not_faulty)
os.mkdir(image_folder + "val")
val_faulty = image_folder + "val/faulty/"
os.mkdir(val_faulty)
val_not_faulty = image_folder + "val/not_faulty/"
os.mkdir(val_not_faulty)
os.mkdir(image_folder + "test")
test_faulty = image_folder + "test/faulty/"
os.mkdir(test_faulty)
test_not_faulty = image_folder + "test/not_faulty/"
os.mkdir(test_not_faulty)


# move images into train, val, test folders
a = 0
b = 0
c = 0
for i in range(total_num):
    file_list = os.listdir(faulty)
    #print(file_list)
    f1 = random.choice(file_list)
    #print(f1)
    if a < total_num * 0.8:
        os.rename(faulty + f1, train_faulty + f1)
        os.rename(not_faulty + f1, train_not_faulty + f1)
        a = a + 1
    elif b < total_num * 0.1:
        os.rename(faulty + f1, val_faulty + f1)
        os.rename(not_faulty + f1, val_not_faulty + f1)
        b = b + 1
    else:
        os.rename(faulty + f1, test_faulty + f1)
        os.rename(not_faulty + f1, test_not_faulty + f1)
        c = c + 1

os.rmdir(faulty)
os.rmdir(not_faulty)