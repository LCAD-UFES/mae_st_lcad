import os, random, unittest
import numpy as np, torch
from tests import _package  # noqa: F401
from tests._videos import decoded, write_dataset
from mae_st.util.kinetics import Kinetics

T, H, W = 24, 16, 120
MEAN, STD = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)

def normalized(raw): return ((torch.from_numpy(raw).float() / 255 - torch.tensor(MEAN)) / torch.tensor(STD)).permute(3, 0, 1, 2)

class DatasetCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.root = write_dataset([rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8) for _ in range(2)])
        cls.raw = decoded(os.path.join(cls.root, "video_000.mp4"))

    def setUp(self): random.seed(0); np.random.seed(0); torch.manual_seed(0)

    def dataset(self, **kw):
        args = dict(mode="pretrain", path_to_data_dir=self.root, sampling_rate=1, num_frames=8, target_fps=10, train_crop_size=(H, W), test_crop_size=(H, W), train_random_horizontal_flip=False)
        return Kinetics(**{**args, **kw})

    def window(self, clip, crop_w):
        # the sample must be an un-interpolated window of consecutive native frames
        for t0 in range(T - clip.shape[1] + 1):
            for j in range(W - crop_w + 1):
                if torch.equal(clip, normalized(self.raw[t0 : t0 + clip.shape[1], :, j : j + crop_w])): return t0, j
        return None

class ResampleTest(DatasetCase):
    def test_default_does_not_resample(self):
        clip = self.dataset()[0][0][0]
        self.assertEqual(tuple(clip.shape), (3, 8, H, W))
        self.assertIsNotNone(self.window(clip, W))

    def test_default_crops_a_native_resolution_window(self):
        clip = self.dataset(train_crop_size=(H, 40))[0][0][0]
        self.assertIsNotNone(self.window(clip, 40))

    def test_resampling_can_be_turned_back_on(self):
        clip = self.dataset(spatial_resample=True, train_jitter_scales=(16, 20), train_crop_size=16)[0][0][0]
        self.assertEqual(tuple(clip.shape), (3, 8, 16, 16))

    def test_rand_aug_requires_resampling(self):
        with self.assertRaises(ValueError): self.dataset(mode="finetune", rand_aug=True)

class SampleIndexTest(DatasetCase):
    def test_fixed_indices_take_the_first_clip_and_the_centre_window(self):
        clip = self.dataset(train_crop_size=(H, 40), temporal_sample_index=0, spatial_sample_index=1)[0][0][0]
        self.assertEqual(self.window(clip, 40), (0, 40))

    def test_fixed_indices_are_repeatable(self):
        ds = self.dataset(train_crop_size=(H, 40), temporal_sample_index=0, spatial_sample_index=1)
        self.assertTrue(torch.equal(ds[0][0], ds[0][0]))

    def test_default_indices_are_random(self):
        ds = self.dataset(train_crop_size=(H, 40))
        self.assertGreater(len({self.window(ds[0][0][0], 40) for _ in range(6)}), 1)

class NormalizationTest(DatasetCase):
    def test_rand_aug_path_uses_the_configured_mean_and_std(self):
        from unittest import mock
        from mae_st.util.decoder import utils
        seen = []
        real = utils.tensor_normalize
        with mock.patch.object(utils, "tensor_normalize", side_effect=lambda f, m, s: seen.append((tuple(m), tuple(s))) or real(f, m, s)):
            self.dataset(mode="finetune", rand_aug=True, spatial_resample=True, train_jitter_scales=(16, 20), train_crop_size=16, mean=(0.1, 0.2, 0.3), std=(1.0, 2.0, 3.0))[0]
        self.assertEqual(set(seen), {((0.1, 0.2, 0.3), (1.0, 2.0, 3.0))})

if __name__ == "__main__": unittest.main()

class CsvFileTest(DatasetCase):
    def test_default_reads_the_csv_of_the_mode(self):
        self.assertEqual(len(self.dataset()._path_to_videos), 2)

    def test_named_csv_replaces_the_csv_of_the_mode(self):
        video = os.path.join(self.root, "video_001.mp4")
        with open(os.path.join(self.root, "slice.csv"), "w") as f: f.write(f"{video} 0\n")
        self.assertEqual(self.dataset(csv_file="slice.csv")._path_to_videos, [video])
