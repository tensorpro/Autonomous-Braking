from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

m = .7
b = -.1
bl = -.1
br = .1
sh = .13

def show_ped(image, bb):
    im = np.zeros(image.shape[:2])
    [ymin, xmin, ymax, xmax] = bb
    im[ymin:ymax,xmin:xmax]=1
    plt.imshow(im)
    plt.show()
    
def in_region(x,y, m=0, b=0, above=True):
    return ((m*x+b) <= y) == above

def y_intercept(x,y,m):
    return y-m*x

def upper_left(bb, m=m, b=bl):
    [ymin, xmin, ymax, xmax] = bb
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    return in_region(x,y,m,b, above=False)

def upper_right(bb, m=m,b=br):
    [ymin, xmin, ymax, xmax] = bb
    yh = .2
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    b_ = y_intercept(1,b, -m)
    return in_region(x,y,-m,b_, above=False)

def brakezone(bb):
    calls = [upper_right, upper_left]
    should_brake = ([c(bb) for c in calls])
    return all(should_brake)

def safezone(bb, sh=sh):
    [ymin, xmin, ymax, xmax] = bb
    x = (xmin+xmax)/2
    y = (ymin+ymax)/2
    return in_region(x,y, b=sh)

def in_trajectory(bb):
    return brakezone(bb) and not safezone(bb)

def show_brakezone(size=[480,480], width=20, height=40, brake_fn=brakezone,
                   safe_fn=safezone, img=None):
    
    if img is not None:
        out = img.copy()
        size = img.shape[:2]
    img_h, img_w = size
    zone = np.zeros(size)
    dx = width/2
    dy = height/2
    for y_ in range(img_h):
        for x_ in range(img_w):
            y = 1-y_/img_h
            x = x_/img_w
            bb = [y-dy,x-dx,y+dy,x+dx]
            info = [(15, None, bb)]
            brake = int(brake_fn(bb) and not safe_fn(bb))
            zone[y_,x_]=brake
            if img is not None and brake:
                out[y_,x_][1]+=40
                # img[0] = max(img[0], 254)
                
    
    # plt.imshow(zone)
    # plt.savefig("killzone")
    # plt.show()
    # plt.close()
    plt.imshow(out)
    plt.savefig("mask")
    plt.show()
