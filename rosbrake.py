import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from automation import Automation
import rospy
import copy
import cv2
from std_msgs.msg import Float32
from brake import in_trajectory
ready = False
image = None

from modules import SSD, YOLO

detection = 'SSD'

def load_yolo():
    return YOLO("model_files/yolo.weights", "model_files/yolo.cfg")

def load_ssd():
    return SSD("model_files/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt")

if detection == 'SSD':
    detect = load_ssd()
else:
    detect = load_yolo()

import time

def to_bb(res_dict):
    h, w = img.shape[:2]
    xmin = res['upperleft']['x']/w
    xmax = res['bottomright']['x']/w
    ymin = res['upperleft']['y']/h
    ymax = res['bottomright']['y']/h
    return [ymin, xmin, ymax, xmax]

def should_brake(img, res):
    brake = []
    for r in res:
        if r['label'] == 'person':
            print("Person found")
            bb = to_bb(r)
            brake.append(in_trajectory(bb))
    return any(brake)


def callback(img):
    global brakepub
    start = time.time()
    res = detect(img)
    print("Time to detect {}".format(time.time() - start))
    print(res)
    brake = should_brake(img, res)
    braketime = Float32()
    braketime.data = 5.0
    if brake:
        print("Braking!")
        brakepub.publish(braketime)
        

count = 0
brakepub = None

if __name__ == "__main__":
    automation = Automation()
    brakepub = rospy.Publisher("/automation/stop", Float32, queue_size=1)
    automation.OnImage(callback)

    rate = rospy.Rate(60)
    count = 0
    while not rospy.is_shutdown():
	# Do some work
        rate.sleep()
