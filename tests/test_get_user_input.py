import unittest
from io import StringIO
from unittest.mock import patch
import pytest

from assesSEM.get_user_input import get_ok_for_overwrite, \
    get_folder_names


@pytest.mark.parametrize("response, expected", [
    ("", True),
    ("Y", True),
    ("y", True)
])
def test_get_ok_for_overwrite_yes(response, expected):
    with patch('builtins.input', return_value=response):
        overwrite_ok = get_ok_for_overwrite()
        assert overwrite_ok == expected

@pytest.mark.parametrize("response, printed_infos", [
    ("N", 'Aborting...\n'),
    ("n", 'Aborting...\n'),
    ("..5..", "Unexpected input. Input should be either 'y' or 'n'. Aborting")
])
def test_get_ok_for_overwrite_no(response, printed_infos):
    with pytest.raises(SystemExit) as e:
        with patch('sys.stdout', new = StringIO()) as fake_out:
            with patch('builtins.input', return_value='N'):
                get_ok_for_overwrite()
    assert e.type == SystemExit
    assert e.value.code is None
    assert fake_out.getvalue() == printed_infos


@pytest.mark.parametrize("response, expected", [
    ("1", ['dataset1']),
    ("2", ['dataset2']),
    ("3", ['dataset3']),
    ("4", ['dataset4']),
    ("0", ['dataset1', 'dataset2', 'dataset3', 'dataset4'])
])
def test_get_folder_names(response, expected):
    with patch('builtins.input', return_value=response):
        folder_names = get_folder_names()
        assert folder_names == expected


if __name__ == '__main__':
    unittest.main()
