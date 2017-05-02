from __future__ import division


import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
slim = tf.contrib.slim
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from visualizations import show_bboxes

def dump(arr, newfile):
    with open(newfile, 'w') as f:
        pickle.dump(arr, f)

def height(ymax, ymin, h=350,c=3.2):
    return abs((h-ymin)/(ymax-ymin)*c)

def add_height_labels(res):
    for r in res:
        ymax=r['bottomright']['y']
        ymin=r['topleft']['y']
        print("ymax, ymin", (ymax,ymin))
        h = height()
        r['label']+=" {}".format(h)

def save_heights(detect):
    load_name = r"/media/kathrada/My Passport/CleanData/"
    save_name = r"/media/kathrada/My Passport/Heights/"
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    for fn_ in sorted(os.listdir(load_name),key=int):
        fn = os.path.join(run_nb,fn)
        print(fn)
        img = misc.imread(fn, mode='RGB')
        res = detect(img)
        add_height_labels(res)
        show_bboxes(res)
        plt.save(os.path.join(save_name, fn_))
    
        
def save_bbs():
    count = 0
    run_nb = r"/media/kathrada/My Passport/CleanData/"
    save_name = r"/media/kathrada/My Passport/Heights/"
    # Do whatever, as long as the node doesn't exit
    classes = []
    scores = []
    bboxes = []
    for fn in sorted(os.listdir(run_nb),key=int):
        print(fn)
	# Do some work
        if True:
            # Do something with the image
            # We'll just write it to a file
            fn = os.path.join(run_nb,fn)
            img = misc.imread(fn, mode='RGB')
            rclasses, rscores, rbboxes = process_image(img)
            classes.append(rclasses)
            scores.append(rscores)
            bboxes.append(rbboxes)
    dump(bboxes, "bboxes")
    dump(classes, "classes")
    dump(scores, "scores")
