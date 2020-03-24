#! /usr/bin/env python

import os
import cv2
import time
import copy
import json
import warnings
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from db_cam_helper import *
from utils.bbox import draw_boxes
from utils.bbox import draw_box_with_id
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from utils.utils import get_yolo_boxes, makedirs
from object_tracking.deep_sort import nn_matching
from object_tracking.deep_sort.tracker import Tracker
from object_tracking.deep_sort.detection import Detection
from object_tracking.application_util import preprocessing
from object_tracking.application_util import generate_detections as gdet

warnings.filterwarnings("ignore")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto(device_count = {'GPU': 0})
# config.gpu_options.allow_growth = False
tf.keras.backend.set_session(tf.Session(config=config))

def _main_(args):
    ip = False
    config_path = args.conf
    num_cam=None
    if args.count != None:
        num_cam = int(args.count)
    ip_address = args.ipadress.split("-")
    ip_list = []
    engine = generate_db_engine(creds)
    label_loader(engine, label_dict, label_template)
    inference_engine_loader(engine, inference_engine_dict, 1)
    for ip_up in ip_address:
    #    user_psk, ip = ip_up.split("@")
    #    user, psk = user_psk.split(":")
    #    ip_list.append({"ip": ip,"user": user, "psk": psk})
       dump, ipp = ip_up.split("@")
       ip, dump2 = ipp.split(":")
       ip_list.append({"ip":ip})

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    if num_cam == None and len(ip_list) != 0:
        ip = True
        num_cam = len(ip_list)
    else:
        print("No image sources found")
        exit()
    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Set up the Tracker
    ###############################

    #   Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metrics = []
    trackers = []
    for i in range(num_cam):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
        trackers.append(tracker)



    ###############################
    #   Predict bounding boxes
    ###############################
    # if 'webcam' in input_path:  # do detection on the first webcam
    video_readers = []
    violation_trackers = []
            
    for i in range(num_cam):
        if not ip:
            video_reader = cv2.VideoCapture(i)
        else:
            # connection_string = get_camera_stream_uri(ip_list[i]["ip"], user=ip_dict[i]["user"], psk=ip_dict[i]["psk"])
            connection_string = ip_address[i]
            video_reader = cv2.VideoCapture(connection_string)
        video_readers.append(video_reader)
        violation_trackers.append({"violation":False, "start_time":None, "end_time":None})

    # the main loop
    batch_size = num_cam
    current_time = []
    images = []
    while True:
        for i in range(num_cam):
            ret_val, image = video_readers[i].read()
            current_time.append(datetime.now())
            if ret_val == True: images += [image]

        if (len(images) == batch_size) or (ret_val == False and len(images) > 0):

            batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh,
                                         nms_thresh)

            for i in range(len(images)):
                boxs = [[box1.xmin,box1.ymin,box1.xmax-box1.xmin, box1.ymax-box1.ymin] for box1 in batch_boxes[i]]
                features = encoder(images[i], boxs)

                # print(features)
                # score to 1.0 here).
                detections = []
                for j in range(len(boxs)):
                    label = batch_boxes[i][j].label
                    detections.append(Detection(boxs[j], batch_boxes[i][j].c, features[j],label))

                # Call the tracker
                trackers[i].predict()
                trackers[i].update(detections)

                n_without_helmet = 0
                n_with_helmet = 0
                for track in trackers[i].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    if track.label == 2:
                        n_without_helmet += 1

                    if track.label == 1:
                        n_with_helmet += 1

                    if n_without_helmet > 0:
                        if violation_trackers[i]["violation"] == False:
                            violation_trackers[i]["violation"] = True
                            violation_trackers[i]["Start_time"] = current_time[i]
                            filename = f"CAM {i} {current_time[i].strftime('%d-%m-%Y %I:%M:%S %p')}.jpg"
                            data_dict = {}
                            data_dict["video_id"] = -1
                            data_dict["inference_engine_id"] = model_id
                            data_dict["operating_unit_id"] = int("".join(ip_list[i]['ip'].split(".")))
                            data_dict["frame_id"] = filename
                            data_dict["label_id"] = label_dict["VLO"][list(label_dict["VLO"].keys())[0]]
                            data_dict["event_processed_time_zone"] = "IST"
                            data_dict["event_processed_local_time"] = str(current_time[i])
                            data_dict["event_flag"] = 1
                            data_dict["created_date"] = str(current_time[i])
                            data_dict["created_by"] = model_id
                            data_dict["current_flag"] = current_flag
                            data_dict["active_flag"] = active_flag
                            data_dict["delete_flag"] = delete_flag
                            cv2.imwrite("filename", images[i])
                            event_log_dtl_writer(engine, data_dict)
                            print("Violation Started and Logged to file {filename}")
                            # call db log
                    if n_without_helmet == 0:
                        if violation_trackers[i]["violation"] == True and violation_trackers[i]["end_time"] == None:
                           violation_trackers[i]["end_time"] = current_time[i]
                        elif violation_trackers[i]["violation"] == True and current_time[i] - violation_trackers[i]["end_time"] > timedelta(seconds=10):
                            filename = f"CAM {i} {current_time[i].strftime('%d-%m-%Y %I:%M:%S %p')}.jpg"
                            violation_trackers[i]["violation"] = False
                            violation_trackers[i]["Start_time"] = None
                            violation_trackers[i]["end_time"] = None
                            data_dict = {}
                            data_dict["video_id"] = -1
                            data_dict["inference_engine_id"] = model_id
                            data_dict["operating_unit_id"] = int("".join(ip_list[i]['ip'].split(".")))
                            data_dict["frame_id"] = filename
                            data_dict["label_id"] = label_dict["NVL"][list(label_dict["NVL"].keys())[0]]
                            data_dict["event_processed_time_zone"] = "IST"
                            data_dict["event_processed_local_time"] = str(current_time[i])
                            data_dict["event_flag"] = 0
                            data_dict["created_date"] = str(current_time[i])
                            data_dict["created_by"] = model_id
                            data_dict["current_flag"] = current_flag
                            data_dict["active_flag"] = active_flag
                            data_dict["delete_flag"] = delete_flag
                            cv2.imwrite("filename", images[i])
                            event_log_dtl_writer(engine, data_dict)
                            print("Violation Stopped and Logged to file {filename}")
                            # call db log

                    bbox = track.to_tlbr()
                    # print(track.track_id,"+",track.label)
                    draw_box_with_id(images[i], bbox, track.track_id, track.label, config['model']['labels'])

                # for det in detections:
                #     print(det.label)
                #     bbox = det.to_tlbr()
                #     cv2.rectangle(images[i], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                
                # print("CAM "+str(i))
                print("Persons without helmet = " + str(n_without_helmet))
                print("Persons with helmet = " + str(n_with_helmet))
                cv2.imshow('Cam'+str(i), images[i])
            images = []
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-n', '--count', help='number of cameras')
    argparser.add_argument('-ip', '--ipadress', help='ip address, user, pass of the camera \n example user:password@192.168.1.1\nFor multiple cameras\n user:password@192.168.1.1-user:password@192.168.1.2-.-.-.')
    args = argparser.parse_args()
    _main_(args)
