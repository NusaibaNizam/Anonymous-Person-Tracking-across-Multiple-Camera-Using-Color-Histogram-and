# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 23:15:40 2020

@author: HP
"""

import cv2
import time
import numpy as np
# import argparse
import math


# import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser(description='Run keypoint detection')
# parser.add_argument("--device", default="cpu", help="Device to inference on")
# parser.add_argument("--video_file", default="new.mp4", help="Input Video")

# args = parser.parse_args()

def openpose(input_source, device):
    angle_ary_right_leg = []
    angle_ary_left_leg = []
    angle_ary_right_hand = []
    angle_ary_left_hand = []
    angle_ary = []
    frame_ary = []
    used = []
    count=0
    bool=False
    MODE = "MPI"

    if MODE is "COCO":
        protoFile = "pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    elif MODE is "MPI":
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                      [14, 11], [11, 12], [12, 13]]

    inWidth = 368
    inHeight = 368
    threshold = 0.1

    # input_source = args.video_file
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (frame.shape[1], frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")

    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print("Frame Number", int(frame_number))
        if not hasFrame:
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)
        # for i in range(len(points)):
        # print ("keypoint:" ,i, end = " ")
        # print (points[i])
        bool = False
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                if frame_number not in used and (((partA == 8 and partB == 9)) or ((partA == 11 and partB == 12)) or ((partA == 2 and partB == 3)) or ((partA == 5 and partB == 6))):
                    used.append(frame_number)
                    if points[8] and points[9] and points[10]:
                        bool = True
                        deltaX11 = points[8][0] - points[9][0]
                        deltaY11 = points[8][1] - points[9][1]
                        deltaX21 = points[10][0] - points[9][0]
                        deltaY21 = points[10][1] - points[9][1]
                        angle1 = (math.atan2(deltaX11, deltaY11)-math.atan2(deltaX21, deltaY21)) / math.pi * 180
                        angle_ary_right_leg.append(abs(angle1))
                        print("Angle1", abs(angle1))
                    if points[11] and points[12] and points[13]:
                        bool=True
                        deltaX12 = points[11][0] - points[12][0]
                        deltaY12 = points[11][1] - points[12][1]
                        deltaX22 = points[13][0] - points[12][0]
                        deltaY22 = points[13][1] - points[12][1]
                        angle2 = (math.atan2(deltaX12, deltaY12) - math.atan2(deltaX22, deltaY22)) / math.pi * 180
                        angle_ary_left_leg.append(abs(angle2))
                        print("Angle2", abs(angle2))
                    if points[2] and points[3] and points[4]:
                        bool = True
                        deltaX13 = points[2][0] - points[3][0]
                        deltaY13 = points[2][1] - points[3][1]
                        deltaX23 = points[4][0] - points[3][0]
                        deltaY23 = points[4][1] - points[3][1]
                        angle3 = (math.atan2(deltaX13, deltaY13)-math.atan2(deltaX23, deltaY23)) / math.pi * 180
                        angle_ary_right_hand.append(abs(angle3))
                        print("Angle3", abs(angle3))
                    if points[5] and points[6] and points[7]:
                        bool = True
                        deltaX14 = points[5][0] - points[6][0]
                        deltaY14 = points[5][1] - points[6][1]
                        deltaX24 = points[7][0] - points[6][0]
                        deltaY24 = points[7][1] - points[6][1]
                        angle4 = (math.atan2(deltaX14, deltaY14) - math.atan2(deltaX24, deltaY24)) / math.pi * 180
                        angle_ary_left_hand.append(abs(angle4))
                        print("Angle4",abs(angle4))
                    if bool:
                        count+=1
                        bool=False
                        print("count",count)
                frame_ary.append(frame_number)
        angle_ary=[]
        angle_ary.insert(0,angle_ary_right_leg)
        angle_ary.insert(1,angle_ary_left_leg)
        angle_ary.insert(2,angle_ary_right_hand)
        angle_ary.insert(3,angle_ary_left_hand)
        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)

        vid_writer.write(frame)
    return angle_ary

# plt.plot(frame_ary,angle_ary)

# plt.xlabel('Number of video Frames')
# plt.ylabel('Angles')

# plt.title('Right leg angles')

# plt.show()

# vid_writer.release()