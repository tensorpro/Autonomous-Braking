import matplotlib.pyplot as plt
import cv2
colors = {}
import random
import numpy as np
from modules import scale_box

def show_bboxes(img, result, figsize=(10,10), linewidth=1.5):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    # fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    plt.imshow(img)
    height, width, _ = img.shape
    for r in result:
        lab = r['label']
        conf = r['confidence']
        bb = scale_box(r['box'], width, height)
        print(bb)
        if lab not in colors.keys():
            colors[lab] = (random.random(), random.random(), random.random())
        rect = plt.Rectangle((bb.xmin, bb.ymin), bb.xmax - bb.xmin,
                             bb.ymax - bb.ymin, fill=False,
                             edgecolor=colors[lab],
                             linewidth=linewidth)
        plt.gca().add_patch(rect)
        plt.gca().text(bb.xmin, bb.ymin - 2,
                       '{:s} | {:.3f}'.format(lab, conf),
                       bbox=dict(facecolor=colors[lab], alpha=0.5),
                       fontsize=12, color='white')
    return fig

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = (h, w,3)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis = 2 )
    return buf

def plt_data(img, result):
    f = show_bboxes(img, result)
    return fig2data(f)
