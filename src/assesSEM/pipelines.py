from assesSEM.IO import (create_image_predictions_folders,
                         initialize_result_csv, save_image,
                         get_file_names, both_files_exist)
from assesSEM.get_user_input import (get_folder_names,
                                     get_desired_nr_of_images_per_folder,
                                     get_names_for_image_type_folders,
                                     get_common_image_nrs_from_image_types)
from assesSEM.model_manipulation import build_and_load_existing_model
from assesSEM.postprocessing import get_percentage_values_for_labels, get_maximum_likelihood_label_for_each_pixel
from assesSEM.use_cases import predict_from_images, ImageMetaData


def run_original_pipeline(model_name):
    model, nb_classes = build_and_load_existing_model(name=model_name)

    folder_names = get_folder_names()
    nr_of_images_per_folder = get_desired_nr_of_images_per_folder(folder_names)
    predictions_paths = create_image_predictions_folders(folder_names)

    for iFolder, folder in enumerate(folder_names):
        path_folder_bse, path_folder_cl, path_mm = get_names_for_image_type_folders(folder)
        if model_name == 'default' or model_name == "model_mlo_512_512_2.h5" or model_name == "model_mlo_512_512_unshifted.h5":
            common_images = get_common_image_nrs_from_image_types(path_folder_bse, path_folder_cl)
        elif model_name == "model_mlo_512_512_unshifted_mm.h5":
            common_images = get_common_image_nrs_from_image_types(path_folder_bse, path_folder_cl, path_mm)
        else:
            return ValueError

        predictions_path = predictions_paths[iFolder]

        percentage_table = initialize_result_csv(common_images)

        no_samples = nr_of_images_per_folder[iFolder]
        for iSample in range(no_samples):
            im_name = common_images[iSample]
            if model_name == 'default' or model_name == "model_mlo_512_512_2.h5" or model_name == "model_mlo_512_512_unshifted.h5":
                image_paths, output_file_name = get_file_names(im_name, path_folder_bse, path_folder_cl,
                                                               predictions_path)
                image_meta_data = ImageMetaData(im_name=im_name, bse_path=image_paths['image_path_bse'],
                                                cl_path=image_paths['image_path_cl'])
            elif model_name == "model_mlo_512_512_unshifted_mm.h5":
                image_paths, output_file_name = get_file_names(im_name, path_folder_bse, path_folder_cl,
                                                               predictions_path, path_folder_mm=path_mm)
                image_meta_data = ImageMetaData(im_name=im_name, bse_path=image_paths['image_path_bse'],
                                                cl_path=image_paths['image_path_cl'],
                                                mm_path=image_paths['image_path_mm'])
            else:
                return ValueError

            if both_files_exist(image_paths['image_path_bse'], image_paths['image_path_cl']):
                if model_name == "model_mlo_512_512_unshifted_mm.h5" and \
                        not both_files_exist(image_paths['image_path_bse'], image_paths['image_path_mm']):
                    continue
                predictions_for_all_labels = predict_from_images(model, image_meta_data)
                predicted_image = get_maximum_likelihood_label_for_each_pixel(predictions_for_all_labels)

                save_image(predicted_image, output_file_name)

                percentage_table.loc[iSample] = get_percentage_values_for_labels(im_name,
                                                                                 percentage_table.loc[iSample],
                                                                                 predictions_path,
                                                                                 predicted_image)

        percentage_table.to_csv(predictions_path + '/' + 'results_' + folder + '.csv', index=False)
        print('.csv-file saved!')
