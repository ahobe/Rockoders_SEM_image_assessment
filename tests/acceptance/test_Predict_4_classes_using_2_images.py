import numpy as np
from pytest_bdd import scenario, given, when, then, parsers

from importlib.resources import files

from assesSEM.model_manipulation import build_and_load_existing_model
from assesSEM.postprocessing import get_maximum_likelihood_label_for_each_pixel
from assesSEM.predictors import predict_image_with_slicing
from assesSEM.use_cases import ImageMetaData, predict_from_images


@scenario("../acceptance/Predict_4_classes_using_2_images.feature", "Predicted image from BSE and CL")
def test_predict_4_classes_using_2_images():
    pass


@given("I have a BSE image and a CL image", target_fixture="image_meta_data")
def image_meta_data():
    im_path_BSE = files('assesSEM.test_images').joinpath("BSE_image6_18_1.tif")
    im_path_CL = files('assesSEM.test_images').joinpath("CL_image6_18_1.tif")
    image_meta_data = ImageMetaData(im_name="image6_18_1_delete_after_adding_data",
                                    bse_path=str(im_path_BSE), cl_path=str(im_path_CL))
    return image_meta_data


@given(parsers.parse("I have a model \"{model_name}\""), target_fixture="model")
def model(model_name):
    if model_name == 'default':
        name = "model_mlo_512_512_2.h5"
    elif model_name == "unshifted_mm":
        name = "model_mlo_512_512_unshifted_mm.h5"
    else:
        raise ValueError
    model = build_and_load_existing_model(name=name)
    return model


@when('I use "predict_from_images" and "get_maximum_likelihood_label_for_each_pixel"', target_fixture="predicted_image")
def predicted_image(model, image_meta_data):
    # predictions_for_all_labels = predict_from_images(model, image_meta_data)
    predictions_for_all_labels = predict_from_images(model, image_meta_data, use_predictor=predict_image_with_slicing)
    predicted_image = get_maximum_likelihood_label_for_each_pixel(predictions_for_all_labels)

    return predicted_image


@then("I get a predicted image")
def check_predicted_image(predicted_image):
    assert isinstance(predicted_image, np.ndarray)


@then("the predicted image has 4 classes")
def image_has_4_classes(predicted_image):
    assert np.round(np.count_nonzero(predicted_image == 4)) > 0
    assert np.round(np.count_nonzero(predicted_image == 3)) > 0
    assert np.round(np.count_nonzero(predicted_image == 2)) > 0
    assert np.round(np.count_nonzero(predicted_image == 1)) > 0
    assert len(np.unique(predicted_image)) == 4


@when('I use "get_percentage_values_for_labels"')
def step_impl():
    raise NotImplementedError(u'STEP: When I use "get_percentage_values_for_labels"')


@then("I get percentage values for all 4 labels")
def step_impl():
    raise NotImplementedError(u'STEP: Then I get percentage values for all 4 labels')


@given("I have a BSE image and a CL image and an MM image")
def step_impl():
    raise NotImplementedError(u'STEP: Given I have a BSE image and a CL image and an MM image')


@scenario("../acceptance/Predict_4_classes_using_2_images.feature", "Predicted image from BSE and CL and MM")
def test_predict_4_classes_using_3_images():
    pass

@given("I have a BSE image and a CL image and an MM image", target_fixture="image_meta_data")
def image_meta_data():
    im_path_BSE = files('assesSEM.test_images').joinpath("BSE_image6_18_1.tif")
    im_path_CL = files('assesSEM.test_images').joinpath("CL_image6_18_1.tif")
    im_path_MM = files('assesSEM.test_images').joinpath("MM_image6_18_1.tif")
    image_meta_data = ImageMetaData(im_name="image6_18_1_delete_after_adding_data",
                                    bse_path=str(im_path_BSE), cl_path=str(im_path_CL),
                                    mm_path=str(im_path_MM))
    return image_meta_data

