import argparse, unittest
import torch
from tests import _package  # noqa: F401
from tests.test_models import small
from mae_st import engine_pretrain
from mae_st.util import misc

class AfterIterTest(unittest.TestCase):
    def test_callback_receives_global_iterations_across_epochs(self):
        torch.manual_seed(0)
        model = small(img_size=(16, 60), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4)
        loader = [(torch.randn(1, 3, 4, 16, 60), None) for _ in range(3)]
        args = argparse.Namespace(accum_iter=1, mask_ratio=0.5, clip_grad=None, repeat_aug=1, num_checkpoint_del=0, lr=1e-4, min_lr=1e-4, warmup_epochs=0, epochs=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        seen = []
        for epoch in range(2):
            engine_pretrain.train_one_epoch(model, loader, optimizer, torch.device("cpu"), epoch, misc.NativeScalerWithGradNormCount(fp32=True), args=args, fp32=True, after_iter=seen.append)
        self.assertEqual(seen, list(range(1, 7)))

if __name__ == "__main__": unittest.main()
