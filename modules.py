import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
slim = tf.contrib.slim

import sys

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from darkflow.net.build import TFNet


classes = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


def convert_result(rclasses, rscores, rbboxes, w, h):
    results = []
    for classid, score, bb in zip(rclasses, rscores, rbboxes):
        bb[[0,2]]*=h
        bb[[1,3]]*=w
        bb = bb.astype(int)
        ymin, xmin, ymax, xmax = bb
        res = {}
        res['bottomright'] = dict(x=xmax, y=ymax)
        res['topleft'] = dict(x=xmin, y=ymin)
        res['confidence'] = score
        res['label'] = classes[classid-1]
        results.append(res)
    return results

class SSD:
    def __init__(self, weights = '../checkpoints/ssd_300_vgg.ckpt', mem_frac=1):
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)

        self.sess = sess =tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = isess = tf.InteractiveSession(config=config)
        self.net_shape = net_shape = (300, 300)
        data_format = 'NHWC'
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, weights)

        # SSD default anchor boxes.
        self.ssd_anchors = ssd_net.anchors(net_shape)
        self.bbox_img = bbox_img
        self.img_input = img_input
        self.image_4d = image_4d
        self.ssd_net = ssd_net
        self.predictions = predictions
        self.localisations = localisations

    def __call__(self,img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        img_input = self.img_input
        image_4d = self.image_4d
        predictions = self.predictions
        localisations = self.localisations
        bbox_img = self.bbox_img
        ssd_anchors = self.ssd_anchors
        isess = self.isess
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        res = {}

        h,w = img.shape[:2]

        return convert_result(rclasses, rscores, rbboxes,w,h)

class YOLO:
    def __init__(self, weights, cfg, mem_frac=1):
        options = {"model": cfg,
                   "load": weights,
                   "threshold": 0.1,
                   "gpu":mem_frac}
        self.net = TFNet(options)

    def __call__(self, img):
        return self.net.return_predict(img)
