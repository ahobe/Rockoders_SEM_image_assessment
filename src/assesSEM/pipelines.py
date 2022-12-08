import os
import numpy as np

from assesSEM.IO import create_image_predictions_folders, initialize_result_csv, \
    save_image
from assesSEM.get_user_input import get_folder_names, get_desired_nr_of_images_per_folder, \
    get_common_image_nrs_from_both_image_types, get_names_for_image_type_folders
from assesSEM.model_manipulation import build_and_load_existing_model
from assesSEM.postprocessing import get_percentage_values_for_labels
from assesSEM.use_cases import predict_from_images, predict_from_images, ImageMetaData


def run_original_pipeline(model_name):
    # initializing solver
    model, nb_classes, im_h = build_and_load_existing_model(name=model_name)
    image_meta_data = ImageMetaData()
    image_meta_data.nb_classes = nb_classes
    image_meta_data.image_height = im_h

    # folder structure manipulation
    folder_names = get_folder_names()
    nr_of_images_per_folder = get_desired_nr_of_images_per_folder(folder_names)
    predictions_paths = create_image_predictions_folders(folder_names)

    for iFolder, folder in enumerate(folder_names):
        # pathfinder for all paths (files, folders, folders to be created, files to be created).
        # class for all name and path types, one instance for each sample
        #
        path_folder_bse, path_folder_cl = get_names_for_image_type_folders(folder)
        images_in_both = get_common_image_nrs_from_both_image_types(path_folder_bse, path_folder_cl)

        predictions_path = predictions_paths[iFolder]

        # initialize class again?
        percentage_table = initialize_result_csv(images_in_both)

        no_samples = nr_of_images_per_folder[iFolder]
        for iSample in range(no_samples):
            # only pipeline here
            # put in all paths for current sample
            # prep input from images
            # predict from input
            # save prediction image yes/no
            # calculate percentages from prediction image yes/no
            # save percentages

            im_name = images_in_both[iSample]

            image_path_bse, image_path_cl, output_file_name = get_file_names(im_name, path_folder_bse, path_folder_cl,
                                                                             predictions_path)
            image_meta_data.image_name = im_name
            image_meta_data.image_path_bse = image_path_bse
            image_meta_data.image_path_cl = image_path_cl

            if both_files_exist(image_path_bse, image_path_cl):
                predictions_for_all_labels = predict_from_images(model, image_meta_data)
                predicted_image = get_maximum_likelihood_label_for_each_pixel(predictions_for_all_labels)

                save_image(predicted_image, output_file_name)

                percentage_table.loc[iSample] = get_percentage_values_for_labels(im_name,
                                                                                 percentage_table.loc[iSample],
                                                                                 predictions_path,
                                                                                 predicted_image)

        percentage_table.to_csv(predictions_path + '/' + 'results_' + folder + '.csv', index=False)
        print('.csv-file saved!')


def get_maximum_likelihood_label_for_each_pixel(predictions_for_all_labels):
    test_argmax = np.argmax(predictions_for_all_labels, axis=2)
    return test_argmax


def get_file_names(im_name, path_folder_bse, path_folder_cl, predictions_path):
    output_file_name = predictions_path + '/' + im_name
    image_path_cl = path_folder_cl + im_name
    image_path_bse = path_folder_bse + im_name
    return image_path_bse, image_path_cl, output_file_name


def both_files_exist(image_path_bse, image_path_cl):
    check1 = os.path.isfile(image_path_cl)
    check2 = os.path.isfile(image_path_bse)
    return check1 and check2
