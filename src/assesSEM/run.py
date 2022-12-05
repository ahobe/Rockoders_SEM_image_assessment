import os
import cv2
import numpy as np

from assesSEM.IO import save_image, get_names_for_image_type_folders, create_image_predictions_folder, \
    initialize_result_csv
from assesSEM.get_user_input import get_folder_names, get_desired_nr_of_images_per_folder, \
    get_common_image_nrs_from_both_image_types
from assesSEM.model_manipulation import build_and_load_existing_model
import time

from assesSEM.plotting import get_cmap
from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing

model, nb_classes, im_h = build_and_load_existing_model(name="model_mlo_512_512_2.h5")

cmap_segmentation = get_cmap()

folder_names = get_folder_names()

nr_of_images_per_folder = get_desired_nr_of_images_per_folder(folder_names)

for iFolder, folder in enumerate(folder_names):
    no_samples = nr_of_images_per_folder[iFolder]
    get_common_image_nrs_from_both_image_types()
    path_folder_bse, path_folder_cl = get_names_for_image_type_folders()
    images_in_both = get_common_image_nrs_from_both_image_types(path_folder_bse, path_folder_cl)

    # im_dummy = cv2.imread(path_folder_cl + onlyfiles_cl[0], 0)

    predictions_path = create_image_predictions_folder()

    # CSV array dummy
    percentage_table = initialize_result_csv(images_in_both)

    # Start segmenting images...
    for iSample in range(no_samples):
        im_name = images_in_both[iSample]
        image_path_cl = path_folder_cl + im_name
        image_path_bse = path_folder_bse + im_name
        check1 = os.path.isfile(image_path_cl)
        check2 = os.path.isfile(image_path_bse)
        if check1 == True and check2 == True:
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

            save_image(test_argmax, cmap_segmentation, predictions_path + '/' + im_name)

            total_area = test_argmax.shape[0] * test_argmax.shape[1]
            percentage_table['path'][iSample] = predictions_path + '/' + im_name
            percentage_table['quartz_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 4) / total_area * 100, 2)
            percentage_table['otherminerals_rel_area'][iSample] = np.round(
                np.count_nonzero(test_argmax == 3) / total_area * 100, 2)
            percentage_table['overgrowth_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 2) / total_area * 100,
                                                                        2)
            percentage_table['pores_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 1) / total_area * 100, 2)
            print('Done and saved!')

    percentage_table.to_csv(predictions_path + '/' + 'results_' + folder + '.csv', index=False)
    print('.csv-file saved!')
