import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

from assesSEM.IO import deal_with_folder_availability, save_image
from assesSEM.get_user_input import get_folder_names
from assesSEM.model_manipulation import build_and_load_existing_model
import time
import pandas as pd

from assesSEM.plotting import get_cmap
from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing

model, nb_classes = build_and_load_existing_model(name="model_mlo_512_512_2.h5")

cmap_segmentation = get_cmap()

folder_names = get_folder_names()

value = input("Please enter # of images loaded per dataset folder:\n")
print(f'You entered {value}')
no_samples = int(value)

for folder in folder_names:

    print('Opening folder', folder, '..')
    path_folder_cl = folder + '/CL/'
    path_folder_bse = folder + '/BSE/'

    onlyfiles_cl = [f for f in listdir(path_folder_cl) if isfile(join(path_folder_cl, f))]
    onlyfiles_bse = [f for f in listdir(path_folder_bse) if isfile(join(path_folder_bse, f))]
    print('Found', len(onlyfiles_cl), 'files in CL folder:')
    print('Found', len(onlyfiles_bse), 'files in BSE folder:')

    im_dummy = cv2.imread(path_folder_cl + onlyfiles_cl[0], 0)

    # Create 'CL segmented' folder
    directory = 'CL_segmented'
    path = os.path.join(folder, directory)
    deal_with_folder_availability(path)

    # CSV array dummy
    dummy_array = np.zeros([len(onlyfiles_cl), 5])
    df = pd.DataFrame(data=dummy_array,
                      columns=['path', 'quartz_rel_area', 'overgrowth_rel_area', 'otherminerals_rel_area',
                               'pores_rel_area'])

    # Start segmenting images...
    for i in range(no_samples):
        check1 = os.path.isfile(path_folder_cl + onlyfiles_cl[i])
        if check1 == True:
            im_name = onlyfiles_cl[i]
            print('Segmenting', im_name, '..')
            cl_im = cv2.imread(path_folder_cl + im_name, 0)
            cl_im = cl_im / 255  # Normalize
            bse_im = cv2.imread(path_folder_bse + im_name, 0)
            bse_im = bse_im / 255  # Normalize

            # Create input for UNet
            X = np.zeros([cl_im.shape[0], cl_im.shape[1], 2])
            X[:, :, 0] = cl_im
            X[:, :, 1] = bse_im

            # Start segmentation (+moving window w. patch size + smoothing)
            start = time.time()
            predictions_smooth = predict_img_with_smooth_windowing(
                X,
                window_size=im_h,
                subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
                nb_classes=nb_classes,
                pred_func=(
                    lambda img_batch_subdiv: model.predict(img_batch_subdiv)
                )
            )
            end = time.time()
            print('execution time:', end - start)

            # Save &/ plot image
            test_argmax = np.argmax(predictions_smooth, axis=2)

            save_image(test_argmax, cmap_segmentation, path + '/' + im_name)

            total_area = test_argmax.shape[0] * test_argmax.shape[1]
            df['path'][i] = path + '/' + im_name
            df['quartz_rel_area'][i] = np.round(np.count_nonzero(test_argmax == 4) / total_area * 100, 2)
            df['otherminerals_rel_area'][i] = np.round(np.count_nonzero(test_argmax == 3) / total_area * 100, 2)
            df['overgrowth_rel_area'][i] = np.round(np.count_nonzero(test_argmax == 2) / total_area * 100, 2)
            df['pores_rel_area'][i] = np.round(np.count_nonzero(test_argmax == 1) / total_area * 100, 2)
            print('Done and saved!')

    df.to_csv(path + '/' + 'results_' + folder + '.csv', index=False)
    print('.csv-file saved!')
