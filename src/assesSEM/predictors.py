import time

import numpy as np

from assesSEM.predicting.slicing import get_slice_bboxes
from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing


def use_predictor_predict_img_with_smooth_windowing(predictor_input, image_meta_data, model):
    # this should depend on the predictor
    start = time.time()
    predictions_smooth = predict_img_with_smooth_windowing(
        predictor_input,
        window_size=image_meta_data.image_height,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=image_meta_data.nb_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict(img_batch_subdiv)
        )
    )
    end = time.time()
    print('execution time:', end - start)
    return predictions_smooth


def use_predictor_predict_image_with_slicing(predictor_input, image_meta_data, model):
    start = time.time()
    predictions_slicing = predict_image_with_slicing(predictor_input, image_meta_data, model)
    end = time.time()
    return predictions_slicing


def predict_image_with_slicing(predictor_input, image_meta_data, model):
    list_slices = get_slice_bboxes(predictor_input.shape[1], predictor_input.shape[0],
                                   slice_height=image_meta_data.image_height, slice_width=image_meta_data.image_height)
    pred_patches = np.zeros([predictor_input.shape[0], predictor_input.shape[1], image_meta_data.nb_classes])
    print(list_slices)
    for i in list_slices:
        x_min = i[0]
        x_max = i[2]
        y_min = i[1]
        y_max = i[3]
        preds = model.predict(np.expand_dims(predictor_input[x_min:x_max, y_min:y_max, :], axis=0))
        pred_patches[x_min:x_max, y_min:y_max, :] = pred_patches[x_min:x_max, y_min:y_max, :] + preds

    return pred_patches
