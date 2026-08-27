# Training config: ViT-L MAE-ST initialized from the published
# video-mae-200x4-nonorm checkpoint, trained on the 10-video Kinetics-400
# slice. Consumed by tools/train.py and tools/compute_loss.py of mae_st_tooling,
# which copy this file verbatim into the work_dir.
#
# A config is executed by the tooling's drivers, which have already put the
# tooling on sys.path: `import bootstrapper` gives the locations
# (EXPERIMENTS_ROOT, CHECKPOINTS_DIR, DATASETS_ROOT, ...) and `custom.*` the
# hooks. Nothing here hardcodes a path.
#
# Ordered sapiens-style, from most likely to need tuning to least likely.

import os.path as osp

import bootstrapper
from custom.viz_hooks import ReconstructionVideoHook

# --- Tier 1: what you tweak run to run --------------------------------------
# Constant learning rate needs the trio below: upstream lr_sched applies
# warmup then half-cycle cosine from lr down to min_lr, unconditionally.
# Equal endpoints and no warmup make the schedule flat.
lr = 1e-5
min_lr = 1e-5
warmup_epochs = 0
blr = None  # base lr (scaled by effective batch / 256); only used when lr is None

mask_ratio = 0.9  # the ratio this checkpoint was pretrained with
max_epochs = 50

# Checkpoint every N epochs; 0 = only the final checkpoint at the end of
# training (NOT recommended for long runs: an interruption loses everything).
save_every_epochs = 5
max_keep_ckpts = 2  # prune older epoch_N.pth beyond this many

# Visualization hooks every N optimizer steps (iterations, not epochs --
# on large datasets an epoch is far too coarse); 0 = never during training
# (use the compute_loss task afterwards). With 10 videos, batch_size=2 and
# drop_last=False there are 5 iterations per epoch, so 100 = every 20 epochs.
visualize_every_iterations = 100

batch_size = 2
repeat_aug = 2   # clips per video per step -> effective 4 clips per step
accum_iter = 1
seed = 0

# --- Tier 2: specific to this test, fixed once set -------------------------
# A bare name is looked up under DATASETS_ROOT (~/datasets by default).
_dataset_common = dict(
    path_to_data_dir="kinetics400_slice10",
    sampling_rate=4,
    num_frames=16,
    train_jitter_scales=(256, 320),
    train_crop_size=224,
)
train_dataset = dict(  # reads train.csv; random clip + random-resized crop + hflip
    mode="pretrain",
    repeat_aug=repeat_aug,
    train_random_horizontal_flip=True,
    jitter_scales_relative=[0.5, 1.0],
    jitter_aspect_relative=[0.75, 1.3333],
    **_dataset_common,
)
# NOTE: in this slice train.csv and val.csv list the SAME 10 videos, so "val"
# here is a fixed CLIP per video (decoded once, fixed masks), not held-out
# videos. Point path_to_data_dir at a real held-out set for a train/val gap.
val_dataset = dict(  # reads val.csv; one clip per video, no flip
    mode="val",
    repeat_aug=1,
    train_random_horizontal_flip=False,
    **_dataset_common,
)
val_max_videos = 10   # fixed clips used for val loss AND for the hooks
val_batch_size = 2
num_workers = 4

weight_decay = 0.05
bias_wd = False
betas = (0.9, 0.95)
clip_grad = None
fp32 = True  # upstream default; no AMP scaler state to worry about

# --- Tier 3: model geometry (change only when switching checkpoints) -------
# Must match video-mae-200x4-nonorm.pth exactly: patch_embed Conv3d kernel
# (t=2, 16x16), joint pos_embed (1, 8*196, 1024), decoder 512-dim x 4 blocks,
# decoder_pred 1536 = 2*16*16*3 outputs. pred_t_dim=16 gives
# t_pred_patch_size = 2*16//16 = 2 (reconstruct all 16 frames), which is
# what makes decoder_pred 1536-wide. No class token, joint pos_embed.
model_name = "mae_vit_large_patch16"
model_kwargs = dict(
    num_frames=16,
    t_patch_size=2,
    decoder_embed_dim=512,
    decoder_depth=4,
    decoder_num_heads=16,
    pred_t_dim=16,
    norm_pix_loss=False,  # how this checkpoint was trained
    sep_pos_embed=False,
    cls_embed=False,
)
# Initial weights when starting from scratch (ignored on --resume); a bare
# name is looked up under EXPERIMENTS_ROOT/checkpoints/. Fused attn.qkv keys
# are split into q/k/v on load (custom/checkpoints.py).
load_from = "video-mae-200x4-nonorm.pth"

# --- Tier 4: conventions/infrastructure ------------------------------------
# work_dir derives from THIS FILE's name:
#   EXPERIMENTS_ROOT/work_dirs/mae_st_overfit_tests/<stem>
work_dir = bootstrapper.work_dir_for(osp.splitext(osp.basename(__file__))[0])

hooks = [
    ReconstructionVideoHook(
        mask_ratio=mask_ratio,
        fps=7.5,  # 16 frames span ~2.1 s of source video (sampling_rate 4 @ 30 fps)
    ),
]
