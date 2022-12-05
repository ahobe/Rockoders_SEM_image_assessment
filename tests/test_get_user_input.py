import unittest
from unittest.mock import Mock, patch
import pytest

from assesSEM.get_user_input import deal_with_folder_availability, \
    get_folder_names


class TestRun(unittest.TestCase):
    def test_deal_with_folder_availability(self):
        self.assertEqual(False, True)


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
