import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import OpenPoseImage
import dom_col_bar_hsv
import cosine_similiarity
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
from random import seed
from random import choice
from random import randint
flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video1', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('video2', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output1', None, 'path to output video')
flags.DEFINE_string('output2', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
    #seed(1)
    value1= randint(0, 999)
    value2= randint(0, 999)
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0


    inputIndex1 = 0
    inputIndex2 = 0
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker1 = Tracker(metric1)
    tracker2 = Tracker(metric2)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    try:
        vid1 = cv2.VideoCapture(int(FLAGS.video1))
        vid2 = cv2.VideoCapture(int(FLAGS.video2))
    except:
        vid1 = cv2.VideoCapture(FLAGS.video1)
        vid2 = cv2.VideoCapture(FLAGS.video2)

    out1 = None
    out2 = None

    if FLAGS.output1 and FLAGS.output2:
        # by default VideoCapture returns float instead of int
        width1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = int(vid1.get(cv2.CAP_PROP_FPS))
        fps2 = int(vid2.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out1 = cv2.VideoWriter(FLAGS.output1, codec, fps1, (width1, height1))
        out2 = cv2.VideoWriter(FLAGS.output2, codec, fps2, (width2, height2))
        list_file1 = open('detection1.txt', 'w')
        list_file2 = open('detection1.txt', 'w')
        frame_index1 = -1
        frame_index2 = -1

    fps1 = 0.0
    fps2 = 0.0
    count1 = 0
    count2 = 0
    similarity = 1
    id1=value1
    id2=value2
    arrHis1 = []
    arrHis2 = []
    arrPose1 = []
    arrPose2 = []
    frame=0;
    while True:
        ret1, img1 = vid1.read()
        ret2, img2 = vid2.read()
        if img1 is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count1 += 1
            if count1 < 3:
                continue
            else:
                break
        if img2 is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count2 += 1
            if count2 < 3:
                continue
            else:
                break
        if frame==46:
            print("if")
            similarity=cosine_similiarity.cos_sim(arrPose1,arrPose2,arrHis1,arrHis2)
            print("compared")
            if similarity==0:
                id1=id2
            elif id1==id2:
                id2=choice([i for i in range(0,999) if i not in [id1]])

        img_in1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img_in2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_in1 = tf.expand_dims(img_in1, 0)
        img_in2 = tf.expand_dims(img_in2, 0)
        img_in1 = transform_images(img_in1, FLAGS.size)
        img_in2 = transform_images(img_in2, FLAGS.size)

        t1 = time.time()
        boxes1, scores1, classes1, nums1 = yolo.predict(img_in1)
        boxes2, scores2, classes2, nums2 = yolo.predict(img_in2)
        classes1 = classes1[0]
        classes2 = classes2[0]
        names = []
        for i in range(len(classes1)):
            names.append(class_names[int(classes1[i])])
        names = np.array(names)
        converted_boxes1 = convert_boxes(img1, boxes1[0])
        features1 = encoder(img1, converted_boxes1)
        detections1 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                       zip(converted_boxes1, scores1[0], names, features1)]

        # initialize color map
        cmap1 = plt.get_cmap('tab20b')
        colors1 = [cmap1(i)[:3] for i in np.linspace(0, 20, 30)]

        # run non-maxima suppresion
        boxs1 = np.array([d.tlwh for d in detections1])
        scores1 = np.array([d.confidence for d in detections1])
        classes1 = np.array([d.class_name for d in detections1])
        indices1 = preprocessing.non_max_suppression(boxs1, classes1, nms_max_overlap, scores1)
        detections1 = [detections1[i] for i in indices1]

        # Call the tracker
        tracker1.predict()
        tracker1.update(detections1)

        for track in tracker1.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox1 = track.to_tlbr()
            class_name1 = track.get_class()
            if class_name1 == "Person" or class_name1 == "person":
                inputIndex1 += 1
                color1 = colors1[int(id1) % len(colors1)]
                color1 = [i * 255 for i in color1]
                cv2.rectangle(img1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), color1, 2)
                #######
                im1 = img1[int(int(bbox1[1])):int(int(bbox1[3])), int(int(bbox1[0])):int(int(bbox1[2]))]
                #######
                cv2.rectangle(img1, (int(bbox1[0]), int(bbox1[1] - 30)),
                              (int(bbox1[0]) + (len(class_name1) + len(str(id1))) * 17, int(bbox1[1])),
                              color1, -1)
                ##############
                # cv2.imwrite("C:\Yolov3DeepSortPersonID\yolov3_deepsort\data\Cropped"+str(inputIndex)+".png", im)
                color1 = ('b', 'g', 'r')
                cv2.putText(img1, class_name1 + "-" + str(id1), (int(bbox1[0]), int(bbox1[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)
                if frame>5 and frame <= 45:
                    hsv_value1 = dom_col_bar_hsv.color_bar(im1)
                    Aflat = np.hstack(hsv_value1)
                    arrHis1.extend(Aflat)
                    arrPose1.extend(OpenPoseImage.openpose(im1,"cpu"))



        names = []
        for i in range(len(classes2)):
            names.append(class_names[int(classes2[i])])
        names = np.array(names)
        converted_boxes2 = convert_boxes(img2, boxes2[0])
        features2 = encoder(img2, converted_boxes2)
        detections2 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                       zip(converted_boxes2, scores2[0], names, features2)]

        # initialize color map
        cmap2 = plt.get_cmap('tab20b')
        colors2 = [cmap2(i)[:3] for i in np.linspace(0, 20, 30)]

        # run non-maxima suppresion
        boxs2 = np.array([d.tlwh for d in detections2])
        scores2 = np.array([d.confidence for d in detections2])
        classes2 = np.array([d.class_name for d in detections2])
        indices2 = preprocessing.non_max_suppression(boxs2, classes2, nms_max_overlap, scores2)
        detections2 = [detections2[i] for i in indices2]

        # Call the tracker
        tracker2.predict()
        tracker2.update(detections2)

        for track in tracker2.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox2 = track.to_tlbr()
            class_name2 = track.get_class()
            if class_name2 == "Person" or class_name2 == "person":
                inputIndex2 += 1
                color2 = colors2[int(id2) % len(colors2)]
                color2 = [i * 255 for i in color2]
                cv2.rectangle(img2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), color2, 2)
                #######
                im2 = img2[int(int(bbox2[1])):int(int(bbox2[3])), int(int(bbox2[0])):int(int(bbox2[2]))]
                #######
                cv2.rectangle(img2, (int(bbox2[0]), int(bbox2[1] - 30)),
                              (int(bbox2[0]) + (len(class_name2) + len(str(id2))) * 17, int(bbox2[1])),
                              color2, -1)
                ##############
                # cv2.imwrite("C:\Yolov3DeepSortPersonID\yolov3_deepsort\data\Cropped"+str(inputIndex)+".png", im)
                color2 = ('b', 'g', 'r')
                cv2.putText(img2, class_name2 + "-" + str(id2), (int(bbox2[0]), int(bbox2[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)
                if frame>5 and frame <= 45:
                    hsv_value2 = dom_col_bar_hsv.color_bar(im2)
                    Bflat = np.hstack(hsv_value2)
                    arrHis2.extend(Bflat)
                    arrPose2.extend(OpenPoseImage.openpose(im2, "cpu"))
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        # for det in detections:
        #    bbox = det.to_tlbr()
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # print fps on screen
        fps1 = (fps1 + (1. / (time.time() - t1))) / 2
        cv2.putText(img1, "FPS: {:.2f}".format(fps1), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        fps2 = (fps2 + (1. / (time.time() - t1))) / 2
        cv2.putText(img2, "FPS: {:.2f}".format(fps2), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output1', 700, 550)
        cv2.imshow('output1', img1)
        cv2.moveWindow('output1', 0, 0)
        cv2.namedWindow('output2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output2', 700, 550)
        cv2.imshow('output2', img2)
        cv2.moveWindow('output2', 700, 0)
        if FLAGS.output1 and FLAGS.output2:
            out1.write(img1)
            out2.write(img2)
            frame_index1 = frame_index1 + 1
            frame_index2 = frame_index2 + 1
            list_file1.write(str(frame_index1) + ' ')
            list_file2.write(str(frame_index1) + ' ')
            if len(converted_boxes1) != 0:
                for i in range(0, len(converted_boxes1)):
                    list_file1.write(str(converted_boxes1[i][0]) + ' ' + str(converted_boxes1[i][1]) + ' ' + str(
                        converted_boxes1[i][2]) + ' ' + str(converted_boxes1[i][3]) + ' ')
            if len(converted_boxes2) != 0:
                for i in range(0, len(converted_boxes2)):
                    list_file2.write(str(converted_boxes2[i][0]) + ' ' + str(converted_boxes2[i][1]) + ' ' + str(
                        converted_boxes2[i][2]) + ' ' + str(converted_boxes2[i][3]) + ' ')
            list_file1.write('\n')
            list_file2.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
        print("frame : ",frame)
        print("ID1 : ",id1)
        print("ID2 : ",id2)
        frame = frame + 1
    vid1.release()
    vid2.release()
    if FLAGS.ouput1 and FLAGS.ouput2:
        out1.release()
        out2.release()
        list_file1.close()
        list_file2.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
