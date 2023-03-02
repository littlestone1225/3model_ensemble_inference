#!/usr/bin/env python3
import os, sys
from datetime import datetime
import cv2
from operator import add
import torch
import time
import logging
import math
import numpy as np
from multiprocessing import Process, Queue, Pipe
import SharedArray


aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

sys.path.append(os.path.join(aoi_dir, "config"))
from config import read_config_yaml

sys.path.append(os.path.join(aoi_dir, "utils"))
from utils import os_makedirs, shutil_move
from logger import get_logger

sys.path.append(os.path.join(aoi_dir, "data_preprocess"))
from remove_black_border import remove_border
from crop_small_image import crop_sliding_window, clahe_transfer
from csv_json_conversion import csv_to_json, json_to_bbox

sys.path.append(os.path.join(aoi_dir, "validation"))
from nms import non_max_suppression_slow
from validation import unify_batch_predictor_output

sys.path.append(os.path.join(aoi_dir, "ensemble"))
from ensemble import ensemble,ensemble_add

sys.path.append(os.path.join(aoi_dir, "YOLOv4"))
import darknet
from darknet_inference import image_detection, batch_detection

sys.path.append(os.path.join(aoi_dir, "detectron2/projects/CenterNet2"))
from centernet.config import add_centernet_config

sys.path.append(os.path.join(aoi_dir, "detectron2"))
from detectron2.config import get_cfg
from detectron2.engine import BatchPredictor

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

# Read config_file
config_file = os.path.join(aoi_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

num_to_category_dict = {0: 'bridge', 1: 'appearance_less', 2: 'excess_solder', 3: 'appearance'}
category_to_color_dict = {'bridge': [0, 0, 255], 'appearance_less': [255,191,0], 'excess_solder': [221,160,221], 'appearance': [0,165,255]}
default_color = [0, 255, 0] # in BGR


# A5000 (8315 MB / 24256 MB)
# centernet2_batch_size = 9 # 2723(MB)
# retinanet_batch_size = 11 # 2995(MB)
# yolov4_batch_size = 1     # 2597(MB)

# # A5000 (7145 MB / 24256 MB) / P4 (4099 MB / 7611 MB)
# centernet2_batch_size = 4 # A5000: 2309(MB) / P4: 1187(MB)
# retinanet_batch_size = 4  # A5000: 2239(MB) / P4: 1241(MB)
# yolov4_batch_size = 1     # A5000: 2597(MB) / P4: 1669(MB)

# P4 (7469 MB / 7611 MB)
# centernet2_batch_size = 16 # 16 => P4: 1999(MB) / 32 => P4: 3067(MB)
# retinanet_batch_size = 24  # P4: 2731(MB) ~ 3577(MB) why?
# yolov4_batch_size = 1      # P4: 1669(MB)

centernet2_batch_size = 18
retinanet_batch_size = 16
yolov4_batch_size = 1

keep_exists = True
use_centernet2 = True
use_retinanet = True
use_yolov4 = True
select_bbox_add_score = True

# https://stackoverflow.com/questions/46802866/how-to-detect-if-the-jpg-jpeg-image-file-is-corruptedincomplete
def check_jpg_integrity(image_file_path):
    for i in range(11):
        with open(image_file_path, 'rb') as f:
            # start_of_image_marker = f.read()[0:2] # b'\xff\xd8'
            end_of_image_marker = f.read()[-2:] # b'\xff\xd9'
            logger.debug("[check_jpg_integrity] image_file_name = {} ; EOI = {}".format
                        (os.path.basename(image_file_path), end_of_image_marker))
            if end_of_image_marker==b'\xff\xd9':
                return True
        if i==10:
            logger.debug("[check_jpg_integrity] fail")
            return False
        else:
            time.sleep(0.5)


def draw_bbox(json_file_name, result_image_dir):
    image_wo_border_dir = config['image_wo_border_dir']
    json_file_path = os.path.join(inference_result_label_dir, json_file_name)
    image_file_name = os.path.splitext(json_file_name)[0] + '.jpg'
    image_file_path = os.path.join(image_wo_border_dir, image_file_name)

    if os.path.isfile(json_file_path):
        bbox_list = json_to_bbox(json_file_path, store_score=True)
    else:
        bbox_list = list()
    image = cv2.imread(image_file_path)

    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = bbox[2:6]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 8)
        '''
        score = bbox[6]
        center_x, center_y = (xmin + xmax)//2, (ymin + ymax)//2
        w, h = xmax-xmin, ymax-ymin
        overlay = image.copy()
        alpha = 0.4
        if score < 0.5:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
        if score > 0.5 and score < 0.8:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        if score > 0.8:
            cv2.ellipse(overlay, (center_x, center_y), (w // 2 + 80, h // 2 + 80), 0, 0, 360, (25, 25, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 0, 255), -1)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
        '''
    result_image_file_path = os.path.join(inference_result_image_dir, image_file_name)
    cv2.imwrite(result_image_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':

    image_dir = config['image_dir']
    image_backup_dir = config['image_backup_dir']
    image_wo_border_dir = config['image_wo_border_dir']

    centernet2_label_dir = config['centernet2_label_dir']
    retinanet_label_dir = config['retinanet_label_dir']
    yolov4_label_dir = config['yolov4_label_dir']
    inference_result_image_dir = config['inference_result_image_dir']
    inference_result_image_backup_dir = config['inference_result_image_backup_dir']
    inference_result_label_dir = config['inference_result_label_dir']
    inference_result_txt_dir = config['inference_result_txt_dir']


    if select_bbox_add_score:
        threshold = {'3hit': 0.2, '2hit': 0.7, '1hit': 0.8}
        inference_result_label_dir = "{}_3{}_2{}_1{}".format(inference_result_label_dir, threshold["3hit"],threshold["2hit"],threshold["1hit"])
        inference_result_image_dir = "{}_3{}_2{}_1{}".format(inference_result_image_dir, threshold["3hit"],threshold["2hit"],threshold["1hit"])
    else:
        center_threshold = {'3hit': 0.1, '2hit': 0.6, '1hit': 0.9}
        retina_threshold = {'3hit': 0.1, '2hit': 0.3, '1hit': 0.9}
        yolov4_threshold = {'3hit': 0.01, '2hit': 0.1, '1hit': 0.6}
        threshold = [center_threshold, retina_threshold, yolov4_threshold] 
        inference_result_image_dir = "{}_3hits".format(inference_result_image_dir)
        inference_result_label_dir = "{}_3hits".format(inference_result_label_dir)

    os_makedirs(inference_result_image_dir, keep_exists)
    os_makedirs(inference_result_label_dir, keep_exists)

    json_list = []
    for img_file_name in os.listdir(image_wo_border_dir):
        file_name, ext = os.path.splitext(img_file_name)
        if ext == ".jpg":
            json_list.append(file_name+".json")
    
    #json_list = ["LKA7627319_LGL2330DA_9691512900U_70.json","LKA8772097_LGL7300DA_9691512900U_130.json"]
    for json_idx, json_file_name in enumerate(json_list):
        
        start_time = time.time()
        if select_bbox_add_score:
            ensemble_add(json_idx, image_wo_border_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                    inference_result_label_dir, json_file_name, threshold, dashboard_txt_dir=inference_result_txt_dir)
        else:
            ensemble(json_idx, image_wo_border_dir, centernet2_label_dir, retinanet_label_dir, yolov4_label_dir, \
                    inference_result_label_dir, json_file_name, threshold, dashboard_txt_dir=inference_result_txt_dir)
        
        logger.info("run_ensemble time = {:4.3f} s".format(round(time.time()-start_time, 3)))

        image_dir = config['image_dir']
        image_backup_dir = config['image_backup_dir']

        end_time = time.time()
        logger.info("ensemble time measure= {:4.3f} s".format(round(end_time-start_time, 3)))

        draw_bbox(json_file_name, inference_result_image_dir)


