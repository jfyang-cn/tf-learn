import cv2
import numpy as np

def readfile_to_dict(filename):
    'Read text file and return it as dictionary'
    d = {}
    f = open(filename)
    for line in f:
        # print(str(line))
        if line != '\n':
            (key, val) = line.split()
            key = key.replace('\\','/')
            d[key] = int(val)

    return d

def calculateRGBdiff(sequence_img):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)
    # find RGBdiff frame 2nd to last frame
    for i in range(length-1,0,-1): # count down
        sequence_img[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])

#     sequence_img[0] = sequence_img[1]
#     np.delete(sequence_img, 0, axis=0)    # remove first image
    return sequence_img