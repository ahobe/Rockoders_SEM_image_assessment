import numpy as np
from pytest_bdd import scenario, given, when, then

from importlib.resources import files

from assesSEM.model_manipulation import build_and_load_existing_model
from assesSEM.use_cases import ImageMetaData, predict_from_images


@scenario("../acceptance/Predict_4_classes_using_2_images.feature", "Predicted image from BSE and CL")
def test_predict_4_classes_using_2_images():
    pass


@given("I have a BSE image and a CL image", target_fixture="image_meta_data")
def image_meta_data():
    im_path = files('assesSEM.test_images').joinpath("image6_18_1_delete_after_adding_data.tif")
    im_path = str(im_path)
    image_meta_data = ImageMetaData(classes_nr=5, im_h=512, im_name="image6_18_1_delete_after_adding_data",
                                    bse_path=im_path, cl_path=im_path)
    return image_meta_data


@given("I have a model", target_fixture="model")
def model():
    model, _, _ = build_and_load_existing_model(name="model_mlo_512_512_2.h5")
    return model


@when('I use "predict_from_images"', target_fixture="predicted_image")
@when("I have a predicted image")
def predicted_image(model, image_meta_data):
    predicted_image = predict_from_images(model, image_meta_data)
    return predicted_image


@then("I have a predicted image")
def check_predicted_image(predicted_image):
    assert int(np.unique(predicted_image)[0]) == 0
