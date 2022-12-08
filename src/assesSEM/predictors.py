import time

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