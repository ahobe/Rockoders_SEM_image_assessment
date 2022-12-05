import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from assesSEM.get_user_input import deal_with_folder_availability


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


def get_names_for_image_type_folders(folder):
    print('Opening folder', folder, '..')
    path_folder_cl = folder + '/CL/'
    path_folder_bse = folder + '/BSE/'
    return path_folder_bse, path_folder_cl


def create_image_predictions_folder(folder):
    new_directory = 'CL_segmented'
    path = os.path.join(folder, new_directory)
    deal_with_folder_availability(path)
    return path


def initialize_result_csv(files_cl):
    col_names = ['path', 'quartz_rel_area', 'overgrowth_rel_area', 'otherminerals_rel_area',
                 'pores_rel_area']
    dummy_array = np.zeros([len(files_cl), len(col_names)])
    df = pd.DataFrame(data=dummy_array, columns=col_names)
    return df


def read_and_normalize_image(image_path):
    im = cv2.imread(image_path, 0)
    im = im / 255  # Normalize
    return im