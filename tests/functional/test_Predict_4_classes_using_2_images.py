from pytest_bdd import scenario, given, when, then

from importlib.resources import files
from assesSEM.use_cases import ImageMetaData


@scenario("../acceptance/Predict_4_classes_using_2_images.feature", "Predicted image from BSE and CL")
def test_predict_4_classes_using_2_images():
    pass


@given("I have a BSE image and a CL image", target_fixture="image_meta_data")
def have_two_images():
    im_path = files('assesSEM.test_images').joinpath("image6_18_1_delete_after_adding_data.tiff")
    image_meta_data = ImageMetaData(classes_nr=5, im_h=512, im_name="image6_18_1_delete_after_adding_data",
                                    bse_path=im_path, cl_path=im_path)
    return image_meta_data


@given("I have a model")
def step_impl():
    raise NotImplementedError(u'STEP: And I have a model')


@when('I use "predict_from_images"')
def step_impl():
    raise NotImplementedError(u'STEP: When I use "predict_from_images"')


@then("I get a predicted image")
def step_impl():
    raise NotImplementedError(u'STEP: Then I get a predicted image')
