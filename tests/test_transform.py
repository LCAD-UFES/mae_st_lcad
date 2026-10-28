import importlib.util, random, unittest
import numpy as np, torch
from tests import _package  # noqa: F401
from mae_st.util.decoder import transform, utils

def frames(t=3, h=16, w=60): return torch.arange(t * 3 * h * w, dtype=torch.float32).reshape(t, 3, h, w)

class RectangularCropTest(unittest.TestCase):
    def test_random_crop_takes_a_window_of_the_requested_size(self):
        np.random.seed(0); x = frames(); y = transform.random_crop(x, (16, 20))
        self.assertEqual(tuple(y.shape), (3, 3, 16, 20))
        j = int(y[0, 0, 0, 0] - x[0, 0, 0, 0])
        self.assertTrue(torch.equal(y, x[:, :, :, j : j + 20]))

    def test_uniform_crop_positions_follow_the_longer_axis(self):
        x = frames(w=60)
        for idx, j in ((0, 0), (1, 20), (2, 40)):
            with self.subTest(spatial_idx=idx): self.assertTrue(torch.equal(transform.uniform_crop(x, (16, 20), idx), x[:, :, :, j : j + 20]))

    def test_full_frame_crop_is_the_identity(self):
        x = frames()
        self.assertTrue(torch.equal(transform.random_crop(x, (16, 60)), x))
        self.assertTrue(torch.equal(transform.uniform_crop(x, (16, 60), 1), x))

    def test_int_size_still_crops_a_square(self):
        np.random.seed(0)
        self.assertEqual(tuple(transform.random_crop(frames(h=32), 16).shape[2:]), (16, 16))
        self.assertEqual(tuple(transform.uniform_crop(frames(h=32), 16, 1).shape[2:]), (16, 16))

    def test_a_crop_larger_than_the_frame_is_an_error(self):
        with self.assertRaises(ValueError): transform.random_crop(frames(), (32, 20))
        with self.assertRaises(ValueError): transform.uniform_crop(frames(), (16, 80), 1)

    def test_random_resized_crop_path_produces_the_rectangular_size(self):
        random.seed(0); np.random.seed(0)
        y = utils.spatial_sampling(frames(), -1, crop_size=(16, 20), random_horizontal_flip=False, aspect_ratio=[0.75, 1.33], scale=[0.5, 1.0])
        self.assertEqual(tuple(y.shape[2:]), (16, 20))

if __name__ == "__main__": unittest.main()
