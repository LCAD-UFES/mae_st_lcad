import unittest
import torch
from tests import _package  # noqa: F401
from mae_st import models_mae

def small(**kw):
    geometry = dict(embed_dim=32, depth=1, num_heads=2, decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2, mlp_ratio=2)
    return models_mae.MaskedAutoencoderViT(**{**geometry, **kw})

def square_patchify(imgs, p, u):
    # the upstream formula, which assumed H == W and a square patch
    N, _, T, H, W = imgs.shape; h = w = H // p; t = T // u
    x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
    return torch.einsum("nctuhpwq->nthwupqc", x).reshape(shape=(N, t * h * w, u * p**2 * 3))

class RectangularPatchTest(unittest.TestCase):
    def setUp(self): torch.manual_seed(0)

    def test_round_trip_is_exact(self):
        m = small(img_size=(16, 60), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4)
        x = torch.randn(2, 3, 4, 16, 60)
        self.assertTrue(torch.equal(m.unpatchify(m.patchify(x)), x))

    def test_square_case_matches_the_upstream_formula(self):
        m = small(img_size=32, patch_size=8, num_frames=4, t_patch_size=2, pred_t_dim=4)
        x = torch.randn(2, 3, 4, 32, 32)
        self.assertTrue(torch.equal(m.patchify(x), square_patchify(x, 8, m.t_pred_patch_size)))

    def test_forward_shapes(self):
        m = small(img_size=(16, 60), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4)
        loss, pred, mask = m(torch.randn(2, 3, 4, 16, 60), mask_ratio=0.5)
        tokens = (4 // 2) * (16 // 4) * (60 // 10)
        self.assertEqual((tuple(pred.shape), tuple(mask.shape)), ((2, tokens, 2 * 4 * 10 * 3), (2, tokens)))
        self.assertTrue(torch.isfinite(loss))

    def test_factories_accept_a_patch_size(self):
        m = models_mae.mae_vit_base_patch16(img_size=(16, 60), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4, decoder_depth=1)
        self.assertEqual(tuple(m.patch_embed.patch_size), (4, 10))
        self.assertEqual(tuple(models_mae.mae_vit_base_patch16(img_size=32, num_frames=4, decoder_depth=1).patch_embed.patch_size), (16, 16))

class AttentionImplTest(unittest.TestCase):
    def test_explicit_and_sdpa_agree_and_reach_every_block(self):
        kw = dict(img_size=(16, 60), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4)
        torch.manual_seed(0); a = small(attn_impl="sdpa", **kw).double()
        torch.manual_seed(0); b = small(attn_impl="explicit", **kw).double()
        self.assertEqual({blk.attn.attn_impl for blk in [*b.blocks, *b.decoder_blocks]}, {"explicit"})
        x = torch.randn(2, 3, 4, 16, 60, dtype=torch.double)
        torch.manual_seed(1); la = a(x, mask_ratio=0.5)[0]
        torch.manual_seed(1); lb = b(x, mask_ratio=0.5)[0]
        la.backward(); lb.backward()
        self.assertTrue(torch.allclose(la, lb, rtol=1e-12, atol=0))
        self.assertTrue(torch.allclose(a.blocks[0].attn.q.weight.grad, b.blocks[0].attn.q.weight.grad, rtol=1e-9, atol=1e-12))

    def test_unknown_impl_is_refused(self):
        with self.assertRaises(ValueError): small(img_size=32, patch_size=8, num_frames=4, t_patch_size=2, pred_t_dim=4, attn_impl="flash")

if __name__ == "__main__": unittest.main()
