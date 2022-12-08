import time

from assesSEM.smooth_tiled_predictions import predict_img_with_smooth_windowing
from predictors import use_predictor_predict_img_with_smooth_windowing
from assesSEM.unet import get_unet_input

# Use case should not know which solver it is using.
# It should also not know the specifics of what is required for that solver.
# The input can also be simpler.
# If a predictor is chosen, then the required input depends on the chosen predictor.
#
def get_input_method_for_predictor_and_model(predictor, model):
    # Usage: input_method = get_input_method_for_predictor_and_model(predictor, model)
    #if predictor is instance of predict_img_with_smooth_windowing:
        # if model is instance of unet with size 512:
            # input_method = get_unet_input
    #else:
        # raise ValueError

    return 0


def predict_from_images(model, image_meta_data, use_predictor=use_predictor_predict_img_with_smooth_windowing,
                                 input_method=get_unet_input):
    predictor_input = input_method(image_meta_data.image_path_bse, image_meta_data.image_path_cl)
    predictions_for_each_label = use_predictor(predictor_input, image_meta_data, model)
    return predictions_for_each_label


class ImageMetaData:
    def __init__(self):
        self.image_height = 512
        self.image_name = None
        self.base_dir_path = None
        self.image_path_bse = None
        self.image_path_cl = None
        self.nb_classes = 5


