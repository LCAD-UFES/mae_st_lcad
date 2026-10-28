import os, sys, tempfile, unittest
import av, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrapper  # noqa: F401,E402  (registers the mae_st clone)
from mae_st import models_mae  # noqa: E402
from custom.viz_hooks import ReconstructionVideoHook  # noqa: E402

def model(h, w):
    torch.manual_seed(0)
    return models_mae.MaskedAutoencoderViT(img_size=(h, w), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4, embed_dim=32, depth=1, num_heads=2,
                                           decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2, mlp_ratio=2).eval()

def video_size(path):
    with av.open(path) as c: s = c.streams.video[0]; return s.height, s.width

class LayoutTest(unittest.TestCase):
    def render(self, h, w, **kw):
        run_dir = tempfile.mkdtemp()
        losses = ReconstructionVideoHook(mask_ratio=0.5, **kw)(model(h, w), torch.randn(1, 3, 4, h, w), ["clip"], "eval", run_dir)
        out = os.path.join(run_dir, "vis_data", "eval")
        return losses, video_size(os.path.join(out, os.listdir(out)[0]))

    def test_wide_frames_stack_vertically_at_full_resolution(self):
        losses, size = self.render(16, 60)
        self.assertEqual(size, (64, 60))
        self.assertEqual(set(losses), {"clip"})

    def test_tall_frames_stack_side_by_side(self):
        self.assertEqual(self.render(20, 10)[1], (20, 40))

    def test_layout_can_be_forced(self):
        self.assertEqual(self.render(16, 60, layout="horizontal")[1], (16, 240))
        with self.assertRaises(ValueError): ReconstructionVideoHook(layout="diagonal")

if __name__ == "__main__": unittest.main()
