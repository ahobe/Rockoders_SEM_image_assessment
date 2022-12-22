import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from assesSEM.get_user_input import deal_with_folder_availability
from assesSEM.plotting import get_cmap


def save_image(data, file_name, cmap_name='hackathon'):
    fig, ax, height = plot_prediction(data, cmap_name)
    plt.savefig(file_name, dpi=height)
    plt.close()


def plot_prediction(data, cmap_name='hackathon'):
    if cmap_name == 'hackathon':
        cmap = get_cmap()
    else:
        cmap = cmap_name
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap=cmap, vmin=-1, vmax=4)
    return fig, ax, height


def create_image_predictions_folders(folder_names):
    new_directory = 'CL_segmented'
    paths = []
    for iFolder, folder in enumerate(folder_names):
        path = os.path.join(folder, new_directory)
        deal_with_folder_availability(path)
        paths.append(path)
    return paths


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


def get_file_names(im_name, path_folder_bse, path_folder_cl, predictions_path, path_folder_mm=None):
    output_file_name = predictions_path + '/' + im_name
    image_paths = {'image_path_cl': path_folder_cl + im_name, 'image_path_bse': path_folder_bse + im_name}
    if path_folder_mm:  # not None
        image_paths['image_path_mm'] = path_folder_mm + im_name

    return image_paths, output_file_name


def both_files_exist(image_path_bse, image_path_cl):
    check1 = os.path.isfile(image_path_cl)
    check2 = os.path.isfile(image_path_bse)
    return check1 and check2