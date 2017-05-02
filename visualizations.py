import matplotlib.pyplot as plt

colors = {}

def show_bboxes(img, result, figsize=(10,10), linewidth=1.5):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    width, height, _ = img.shape

    for r in result:
        ymin = r['topleft']['y']
        xmin = r['topleft']['x']
        ymax = r['bottomright']['y']
        xmax = r['bottomright']['x']
        bb = [ymin, xmin, ymax, xmax]
        lab = r['label']
        conf = r['confidence']

        if lab not in colors.keys():
            colors[lab] = (random.random(), random.random(), random.random())
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[lab],
                             linewidth=linewidth)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymin - 2,
                       '{:s} | {:.3f}'.format(lab, conf),
                       bbox=dict(facecolor=colors[lab], alpha=0.5),
                       fontsize=12, color='white')
