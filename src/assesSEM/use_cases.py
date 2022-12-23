from assesSEM.predictors import use_predictor_predict_img_with_smooth_windowing, predict_image_with_slicing
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

# predict_image_with_slicing
def predict_from_images(model, image_meta_data, use_predictor=predict_image_with_slicing,
                                 input_method=get_unet_input):
    predictor_input = input_method(model.name, image_meta_data)
    predictions_for_each_label = use_predictor(predictor_input, image_meta_data, model)
    return predictions_for_each_label


class ImageMetaData:
    def __init__(self, im_h=768, im_w=1024, im_name=None, base_path=None, bse_path=None, cl_path=None, mm_path=None):
        self.image_height = im_h
        self.image_width = im_w
        self.image_name = im_name
        self.base_dir_path = base_path
        self.image_path_bse = bse_path
        self.image_path_cl = cl_path
        self.image_path_mm = mm_path
