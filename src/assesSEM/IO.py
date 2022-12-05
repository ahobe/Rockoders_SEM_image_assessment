import os
import sys

import numpy as np
from matplotlib import pyplot as plt


def deal_with_folder_availability(path):
    if not os.path.exists(path):
        os.mkdir(path)
        empty_folder = True
    else:
        files_in_dir = os.listdir(path)
        if len(files_in_dir) == 0:
            empty_folder = True
        else:
            empty_folder = False

    overwrite_ok = False
    if not empty_folder:
        answer = input("Folder not empty. Is it ok to overwrite? [Y/n]")
        if answer == "Y" or answer == "y" or answer == "":  # only enter was pressed, thus choosing the default
            overwrite_ok = True
        elif answer == "N" or answer == "n":
            print('Aborting...')
            sys.exit()
        else:
            print("Unexpected input. Input should be either 'y' or 'n'. Aborting")
            sys.exit()

    if overwrite_ok or empty_folder:  # either or both True
        allowed_to_continue = True
        return allowed_to_continue


def save_image(data, cm, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, cmap=cm, vmin=-1, vmax=4)
    plt.savefig(fn, dpi=height)
    plt.close()