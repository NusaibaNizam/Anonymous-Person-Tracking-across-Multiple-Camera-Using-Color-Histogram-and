# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:05:07 2020

@author: HP
"""
import cv2
#import time
import dom_col_bar_hsv
import numpy as np
import argparse as ap
#import math
#import matplotlib.pyplot as plt
import video_openpose
from scipy.spatial import distance
import cosine_similiarity

array1=[]
array2=[]

parser = ap.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
#parser.add_argument("--video_file", default="new.mp4", help="Input Video")
parser.add_argument("-v1", "--video1", required = True, help = "Path to the image")
parser.add_argument("-v2", "--video2", required = True, help = "Path to the image")
parser.add_argument("-i1", "--image1", required = True, help = "Path to the image")
parser.add_argument("-i2", "--image2", required = True, help = "Path to the image")

args = parser.parse_args()

img = cv2.imread(args.image1)
img1= cv2.imread(args.image2)

#rgb_value1=[]
#rgb_value2=[]

hsv_value1=dom_col_bar_hsv.color_bar(img)
print(hsv_value1)
hsv_value2=dom_col_bar_hsv.color_bar(img1)
print(hsv_value2)

Aflat = np.hstack(hsv_value1)
Bflat = np.hstack(hsv_value2)
#dist = distance.cosine(rgb_value1, rgb_value2)
dist = distance.cosine(Aflat, Bflat)
if(dist>0):
    similiarity=1-dist
else:
    similiarity=1
#print("Disatnce",dist)
#print ("Similiarity",similiarity)
#input_source = args.video1

array1=video_openpose.openpose(args.video1,args.device)


#input_source = args.video2

array2=video_openpose.openpose(args.video2,args.device)

#max_array=max(len(array1),len(array2))
#padded_array = np.zeros(max_array)
#padded_array1 = np.zeros(max_array)
   
#padded_array[:0,:len(array1)] = array1
#padded_array1[:0,:len(array2)] = array2




cosine_similiarity.cos_sim(array1,array2,dist,similiarity,img,img1)

