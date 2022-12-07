import time

from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing
from assesSEM.unet import get_unet_input

# Use case should not know which solver it is using.
# It should also not know the specifics of what is required for that solver.
# The input can also be simpler.
# If a predictor is chosen, then the required input depends on the chosen predictor.
#
def get_input_method_for_predictor_and_model(predictor, model):
    #if predictor is instance of predict_img_with_smooth_windowing:
        # input_method = get_unet_input
    #else:
        # raise ValueError

    return 0


def use_case_predict_from_images(model, images_and_meta, predictor=predict_img_with_smooth_windowing):
    input_method = get_input_method_for_predictor_and_model(predictor, model)
    predictor_input = input_method(images_and_meta)
    predictions = predictor(predictor_input, model)
    return predictions


def predict_from_images(im_h, im_name, image_path_bse, image_path_cl, nb_classes, model):
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
    return predictions_smooth