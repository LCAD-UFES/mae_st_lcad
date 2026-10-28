import glob, json, os, pathlib, subprocess, sys, tempfile, textwrap, unittest
import av, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T, H, W = 24, 16, 60
FAKE_WANDB = '''
import json, os
class _Run:
    name = "fake"
    def log(self, data, step):
        with open(os.environ["FAKE_WANDB_LOG"], "a") as f: f.write(json.dumps({"step": step, **data}) + "\\n")
    def finish(self): pass
def init(**kw): return _Run()
'''
CONFIG = '''
import os.path as osp
import bootstrapper
from custom.viz_hooks import ReconstructionVideoHook
lr = min_lr = 1e-4; warmup_epochs = 0; mask_ratio = 0.5; max_epochs = 2
save_every_epochs = 1; max_keep_ckpts = 1; visualize_every_iterations = 2
batch_size = 1; accum_iter = {accum}; repeat_aug = 1; seed = 0; num_workers = 0; device = "cpu"
_common = dict(path_to_data_dir="ds", sampling_rate=1, num_frames=4, target_fps=10, train_crop_size=({h}, {w}), test_crop_size=({h}, {w}),
               temporal_sample_index=0, spatial_sample_index=1, train_random_horizontal_flip=False)
train_dataset = dict(mode="pretrain", **_common); val_dataset = dict(mode="val", **_common)
model_name = "MaskedAutoencoderViT"
model_kwargs = dict(img_size=({h}, {w}), patch_size=(4, 10), num_frames=4, t_patch_size=2, pred_t_dim=4, embed_dim=32, depth=1, num_heads=2,
                    decoder_embed_dim=16, decoder_depth=1, decoder_num_heads=2, mlp_ratio=2)
work_dir = bootstrapper.work_dir_for(osp.splitext(osp.basename(__file__))[0])
hooks = [ReconstructionVideoHook(mask_ratio=mask_ratio, fps=10)]
wandb = dict(project="e2e", mode="offline")
'''

def write_video(path):
    rng = np.random.default_rng(len(path))
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=10); s.width, s.height, s.pix_fmt = W, H, "yuv444p"; s.options = {"qp": "0"}
        for _ in range(T): c.mux(s.encode(av.VideoFrame.from_ndarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), format="rgb24")))
        c.mux(s.encode())

class TrainEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.env = dict(os.environ, MAE_ST_EXPERIMENTS=os.path.join(cls.tmp, "exp"), MAE_ST_DATASETS=os.path.join(cls.tmp, "data"),
                       PYTHONPATH=os.path.join(cls.tmp, "fake"), FAKE_WANDB_LOG=os.path.join(cls.tmp, "wandb.jsonl"))
        os.makedirs(os.path.join(cls.tmp, "fake")); pathlib.Path(cls.tmp, "fake", "wandb.py").write_text(FAKE_WANDB)
        ds = os.path.join(cls.tmp, "data", "ds"); os.makedirs(ds)
        for i in range(2): write_video(os.path.join(ds, f"clip_{i}.mp4"))
        cls.run_tool("prepare_dataset.py", ds)
        os.makedirs(os.path.join(cls.tmp, "exp", "configs"))
        for name, accum in (("e2e", 1), ("bad_accum", 5)):
            pathlib.Path(cls.tmp, "exp", "configs", f"{name}.py").write_text(textwrap.dedent(CONFIG.format(accum=accum, h=H, w=W)))

    @classmethod
    def run_tool(cls, tool, *args, check=True):
        return subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), *args], env=cls.env, capture_output=True, text=True, check=check)

    def test_train_then_compute_loss(self):
        self.run_tool("train.py", "e2e.py", "--tensorboard", "off")
        work = os.path.join(self.tmp, "exp", "work_dirs", "mae_st_overfit_tests", "e2e")
        read = lambda p: pathlib.Path(p).read_text()
        scalars = [json.loads(l) for l in read(glob.glob(os.path.join(work, "*", "vis_data", "scalars.json"))[0]).splitlines()]
        self.assertEqual([r["epoch"] for r in scalars], [1, 2])
        self.assertTrue(os.path.isfile(read(os.path.join(work, "last_checkpoint")).strip()))
        clip = sorted(glob.glob(os.path.join(work, "*", "vis_data", "epoch*", "*.mp4")))[0]
        with av.open(clip) as c: self.assertEqual((c.streams.video[0].height, c.streams.video[0].width), (4 * H, W))
        self.assertEqual([json.loads(l)["step"] for l in read(self.env["FAKE_WANDB_LOG"]).splitlines()], [1, 2])
        self.run_tool("compute_loss.py", "e2e.py", "--checkpoint", "last", "--dataset", "ds")
        self.assertEqual(len(glob.glob(os.path.join(work, "compute_loss", "*", "*.json"))), 1)

    def test_accum_iter_larger_than_an_epoch_is_refused(self):
        r = self.run_tool("train.py", "bad_accum.py", "--tensorboard", "off", check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("never step", r.stderr)

if __name__ == "__main__": unittest.main()
