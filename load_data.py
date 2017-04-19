import gzip
import numpy as np
import matplotlib.pyplot as plt
import pickle

def summarize(fname):
    with gzip.open(fname, 'rb',9) as f:
        brakes = []
        peds = []
        pedbrakes = []
        while True:
            try:
                d=pickle.load(f)
                peds.append(len(d['peds']))
                brakes.append(d['brake'])
                pedbrakes.append(peds[-1]*brakes[-1])

            except EOFError:
                break
    plt.title("Pedestrians")
    plt.hist(peds)
    plt.savefig("ped_hist")
    plt.close()

    plt.title("Brakes")
    plt.hist(brakes)
    plt.savefig("brake_hist")
    plt.close()

def show_frame(buff):
    img = np.frombuffer(buff, dtype=np.uint8).reshape(160,320,3)
    plt.imshow(img)
    plt.savefig("lol")
