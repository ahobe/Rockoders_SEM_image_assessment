import os
import time
import numpy as np

from assesSEM.IO import get_names_for_image_type_folders, create_image_predictions_folder, initialize_result_csv, \
    save_image
from assesSEM.get_user_input import get_folder_names, get_desired_nr_of_images_per_folder, \
    get_common_image_nrs_from_both_image_types
from assesSEM.model_manipulation import build_and_load_existing_model
from assesSEM.plotting import get_cmap
from assesSEM.postprocessing import get_percentage_values_for_labels
from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing
from assesSEM.unet import get_unet_input


def run_original_pipeline(model_name):
    model, nb_classes, im_h = build_and_load_existing_model(name=model_name)

    folder_names = get_folder_names()

    nr_of_images_per_folder = get_desired_nr_of_images_per_folder(folder_names)

    for iFolder, folder in enumerate(folder_names):
        no_samples = nr_of_images_per_folder[iFolder]
        get_common_image_nrs_from_both_image_types()
        path_folder_bse, path_folder_cl = get_names_for_image_type_folders()
        images_in_both = get_common_image_nrs_from_both_image_types(path_folder_bse, path_folder_cl)

        # im_dummy = cv2.imread(path_folder_cl + onlyfiles_cl[0], 0)

        predictions_path = create_image_predictions_folder()
        percentage_table = initialize_result_csv(images_in_both)

        for iSample in range(no_samples):
            im_name = images_in_both[iSample]
            image_path_cl = path_folder_cl + im_name
            image_path_bse = path_folder_bse + im_name
            check1 = os.path.isfile(image_path_cl)
            check2 = os.path.isfile(image_path_bse)

            if check1 == True and check2 == True:
                print('Segmenting', im_name, '..')
                unet_input = get_unet_input(image_path_bse, image_path_cl)

                # Start segmentation (+moving window w. patch size + smoothing)
                start = time.time()
                predictions_smooth = predict_img_with_smooth_windowing(
                    unet_input,
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

                cmap_segmentation = get_cmap()
                save_image(test_argmax, cmap_segmentation, predictions_path + '/' + im_name)

                percentage_table = get_percentage_values_for_labels(iSample, im_name, percentage_table, predictions_path, test_argmax)

        percentage_table.to_csv(predictions_path + '/' + 'results_' + folder + '.csv', index=False)
        print('.csv-file saved!')


