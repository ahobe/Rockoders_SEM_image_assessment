import unittest
from io import StringIO
from unittest.mock import patch
import pytest

from assesSEM.get_user_input import (get_ok_for_overwrite,
                                     get_folder_names,
                                     get_desired_nr_of_images_per_folder,
                                     get_model_name_from_user,
                                     get_predictor_name_from_user,
                                     deal_with_folder_availability
                                     )
from assesSEM.predictors import use_predictor_predict_img_with_smooth_windowing, predict_image_with_slicing


@pytest.mark.parametrize("response, expected", [
    ("", True),
    ("Y", True),
    ("y", True)
])
def test_get_ok_for_overwrite_yes(response, expected):
    with patch('builtins.input', return_value=response):
        overwrite_ok = get_ok_for_overwrite('.')
        assert overwrite_ok == expected


@pytest.mark.parametrize("response, printed_infos", [
    ("N", 'Aborting...\n'),
    ("n", 'Aborting...\n'),
    ("..5..", "Unexpected input. Input should be either 'y' or 'n'. Aborting\n")
])
def test_get_ok_for_overwrite_abort(response, printed_infos):
    with pytest.raises(SystemExit) as e:
        with patch('sys.stdout', new=StringIO()) as fake_out:
            with patch('builtins.input', return_value=response):
                get_ok_for_overwrite('.')
    assert e.type == SystemExit
    assert e.value.code is None
    assert fake_out.getvalue() == printed_infos


@pytest.mark.parametrize("response, expected", [
    ("1", ['dataset1']),
    ("2", ['dataset2']),
    ("3", ['dataset3']),
    ("4", ['dataset4A']),
    ("5", ['dataset4B']),
    ("0", ['dataset1', 'dataset2', 'dataset3', 'dataset4A', 'dataset4B'])
])
def test_get_folder_names(response, expected):
    with patch('builtins.input', return_value=response):
        folder_names = get_folder_names()
        assert folder_names == expected


def test_get_folder_names_raises():
    with pytest.raises(ValueError) as e:
        with patch('builtins.input', return_value="..5.."):
            get_folder_names()
    assert e.type == ValueError


@pytest.mark.parametrize("folder_names, response, expected", [
    (['dataset1'], "5", [5]),
    (['dataset2'], "7", [7]),
    (['dataset3'], "12", [12]),
    (['dataset4'], "9", [9]),

])
def test_get_desired_nr_of_images_per_folder(folder_names, response, expected):
    with patch('builtins.input', return_value=response):
        with patch('assesSEM.get_user_input.get_nr_of_images_in_folder', return_value='900'):
            nr_per_folder = get_desired_nr_of_images_per_folder(folder_names)
            assert nr_per_folder == expected


def test_get_desired_nr_of_images_per_folder_multiple():
    folder_names = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
    responses = ["5", "7", "12", "9"]
    expected = [5, 7, 12, 9]
    with patch('builtins.input', side_effect=responses):
        with patch('assesSEM.get_user_input.get_nr_of_images_in_folder', return_value='900'):
            nr_per_folder = get_desired_nr_of_images_per_folder(folder_names)
            assert nr_per_folder == expected


def test_get_desired_nr_of_images_per_folder_raises():
    folder_names = ['stuff']
    with pytest.raises(ValueError) as e:
        with patch('builtins.input', return_value='..5..'):
            get_desired_nr_of_images_per_folder(folder_names)
    assert e.type == ValueError


def test_get_desired_nr_of_images_per_folder_too_many():
    folder_names = ['stuff']
    with patch('builtins.input', return_value='5'):
        with patch('assesSEM.get_user_input.get_nr_of_images_in_folder', return_value='2'):
            nr_per_folder = get_desired_nr_of_images_per_folder(folder_names)
            assert nr_per_folder == [2]


@pytest.mark.parametrize("response, expected", [
    ("1", "model_mlo_512_512_2.h5"),
    ("2", "model_mlo_512_512_unshifted.h5"),
    ("3", "model_mlo_512_512_unshifted_mm.h5"),

])
def test_get_model_name_from_user(response, expected):
    with patch('builtins.input', return_value=response):
        model_name = get_model_name_from_user()
        assert model_name == expected


def test_get_model_name_from_user_raises():
    with pytest.raises(ValueError) as e:
        with patch('builtins.input', return_value="..5.."):
            get_model_name_from_user()
    assert e.type == ValueError


@pytest.mark.parametrize("response, expected", [
    ("1", use_predictor_predict_img_with_smooth_windowing),
    ("2", predict_image_with_slicing),
])
def test_get_predictor_name_from_user(response, expected):
    with patch('builtins.input', return_value=response):
        predictor = get_predictor_name_from_user()
        assert predictor == expected


def test_get_predictor_name_from_user_raises():
    with pytest.raises(ValueError) as e:
        with patch('builtins.input', return_value="..5.."):
            get_predictor_name_from_user()
    assert e.type == ValueError


def test_deal_with_folder_availability_nonexistent():
    with patch('os.mkdir', return_value=1):
        result = deal_with_folder_availability("../tmp")
    assert result == True


def test_deal_with_folder_availability_existent_empty():
    with patch('os.listdir', return_value=[]):
        result = deal_with_folder_availability("..")
    assert result == True


def test_deal_with_folder_availability_raises():
    with pytest.raises(SystemExit) as e:
        with patch('builtins.input', return_value="N"):
            deal_with_folder_availability("..")
    assert e.type == SystemExit
    assert e.value.code is None


if __name__ == '__main__':
    unittest.main()
