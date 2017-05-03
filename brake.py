from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

m = 4
b = -.2
bl = -.1
br = -.1
sh = .13

def show_ped(image, bb):
    im = np.zeros(image.shape[:2])
    [ymin, xmin, ymax, xmax] = bb
    im[ymin:ymax,xmin:xmax]=1
    plt.imshow(im)
    plt.show()
    
def in_region(x,y, m=0, b=0, above=True, from_left=False):
    x = 1 - x if from_left else x
    return ((m*x+b) <= y) == above

def brakezone(x,y,m=m,b=b,sh=.3):
    left = in_region(x,y,m,b, above=False)
    right = in_region(x,y,m,b, above=False, from_left=True)
    top = in_region(x,y,b=.3, above=False)
    return left and right and top

def brake_policy(m=m, b=b,sh=sh):
    def policy(x,y):
        return brakezone(x,y,m=m,b=b, sh=sh)
    return policy

def to_bb(res, img):
    h, w = img.shape[:2]
    xmin = res['topleft']['x']/w
    xmax = res['bottomright']['x']/w
    ymin = res['topleft']['y']/h
    ymax = res['bottomright']['y']/h
    return [ymin, xmin, ymax, xmax]

def res_policy(brake_policy):
    def should_brake(res, in_trajectory=brake_policy):
        brake = []
        for r in res:
            if r['label'] == 'person':
                print("Person found")
                x,y = feet(r)
                brake.append(in_trajectory(x,y))
        return any(brake)
    return should_brake

def feet(res):
    bb = res['box']
    x = (bb.xmax+bb.xmin)/2
    y = bb.ymax
    return x,y
    
def show_brakezone(img, brake_fn=brakezone, saveas=None, show=False):
    if img is None:
        out = np.zeros(size)
    else:
        out = img.copy()
        size = img.shape[:2]
    img_h, img_w = size
    zone = np.zeros(size)
    for y_ in range(img_h):
        for x_ in range(img_w):
            y = 1-y_/img_h
            x = x_/img_w
            brake = brake_fn(x,y) #and not safe_fn(bb)
            zone[y_,x_]=brake
            if img is not None and brake:
                out[y_,x_,0]+=35
                # out[y_,x_,0]=min(200,out[y_,x_][0])
                
    if show:
        plt.imshow(out)
        plt.show()
    if saveas:
        plt.savefig(saveas)
    return out

from visualizations import show_bboxes
    
def find_horizon(img, save="horizon", detect=None, res=None,sh=sh,b=b,m=m):
    if detect:
        res = detect(img)
    sh_in = (raw_input("Enter horizon: "))
    b_in = ( raw_input("Enter Intc: "))
    m_in = ( raw_input("Enter Slope: "))
    update = lambda x, default: float(x) if x is not '' else float(default)
    b = update(b_in, b)
    m = update(m_in, m)
    sh = update(sh_in, sh)
    print('(b,m,sh)',b,m,sh)
    brake_fn=brake_policy(sh=sh, m=m, b=b)
    masked=show_brakezone(img, show=False, brake_fn=brake_fn)
    if detect:
        plt.close()
        res = detect(img)
    if res:
        print(res)
        show_bboxes(masked, res)
        print(res_policy(brake_fn)(res))
    plt.show()

