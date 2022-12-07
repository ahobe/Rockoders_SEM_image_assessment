import time

from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing
from assesSEM.unet import get_unet_input


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