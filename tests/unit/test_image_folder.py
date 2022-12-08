import unittest

from importlib.resources import files

import numpy as np

from assesSEM.IO import read_and_normalize_image

# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here


def test_opencv_reads_tiff():
    im_path = files('assesSEM.test_images').joinpath("image6_18_1_delete_after_adding_data.tif")
    print(type(im_path))
    im = read_and_normalize_image(str(im_path))
    assert int(np.unique(im)[0]) == 0


if __name__ == '__main__':
    unittest.main()
