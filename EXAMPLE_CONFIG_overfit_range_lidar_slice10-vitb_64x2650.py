# Training config: ViT-B MAE-ST from random initialization, overfitting the
# range_lidar_slice10 clips at their full 64 x 2650 resolution (64 beams of
# the Waymo top lidar x 2650 azimuth steps, 10 Hz). Consumed by tools/train.py
# and tools/compute_loss.py, which copy this file verbatim into the work_dir.
#
# Reproduces the settings of the September 2026 range-lidar overfit runs (the
# last of them, at full resolution). The dataset dir needs its csvs written on
# this machine:
#     python tools/prepare_dataset.py <DATASETS_ROOT>/range_lidar_slice10 --min-frames 20
# --min-frames 20 drops the 1-frame trailing clip of the reference set.

import os.path as osp

import bootstrapper
from custom.viz_hooks import ReconstructionVideoHook

# --- Tier 1: what you tweak run to run --------------------------------------
# Warmup then half-cycle cosine from lr to min_lr (upstream lr_sched).
lr = 1e-3
min_lr = 1e-4
warmup_epochs = 5
blr = None

mask_ratio = 0.3
max_epochs = 3000
save_every_epochs = 50
max_keep_ckpts = 2
visualize_every_iterations = 25

# One clip per step, gradients accumulated over the whole set: one optimizer
# step per epoch on the full-set gradient. accum_iter must not exceed the
# iterations per epoch (number of clips / batch_size), or no step ever runs.
batch_size = 1
accum_iter = 9
repeat_aug = 1
seed = 0

# --- Tier 2: specific to this test, fixed once set -------------------------
# Same clips, same window and no augmentation every epoch, so the model can
# memorize them: the first num_frames frames (temporal index 0), the whole
# frame (crop = frame, centre position), no flip. mae_st main never resamples
# range images (spatial_resample=False is its default).
_frame = (64, 2650)
_mean, _std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)  # the runs used the loader defaults; dataset statistics would go here
_dataset_common = dict(
    path_to_data_dir="range_lidar_slice10",
    sampling_rate=1,
    num_frames=20,
    target_fps=10,
    train_crop_size=_frame,
    test_crop_size=_frame,
    temporal_sample_index=0,
    spatial_sample_index=1,
    train_random_horizontal_flip=False,
    mean=_mean,
    std=_std,
)
train_dataset = dict(mode="pretrain", repeat_aug=repeat_aug, **_dataset_common)
val_dataset = dict(mode="val", repeat_aug=1, **_dataset_common)
val_max_videos = None
val_batch_size = 1
num_workers = 4

weight_decay = 0.0
bias_wd = False
betas = (0.9, 0.95)
clip_grad = None
fp32 = False  # mixed precision: the 8480-token sequences are memory-heavy

# --- Tier 3: model geometry ---------------------------------------------------
# Patch 4 beams x 50 azimuth steps; 20 frames in temporal patches of 2:
# a (10, 16, 53) token grid, 8480 tokens. pred_t_dim=20 reconstructs every
# frame (t_pred_patch_size 2), so decoder_pred has 2 * 4 * 50 * 3 = 1200 outputs.
model_name = "mae_vit_base_patch16"
model_kwargs = dict(
    img_size=_frame,
    patch_size=(4, 50),
    num_frames=20,
    t_patch_size=2,
    decoder_embed_dim=512,
    decoder_depth=4,
    pred_t_dim=20,
    norm_pix_loss=True,
)

# --- Tier 4: conventions/infrastructure ------------------------------------
work_dir = bootstrapper.work_dir_for(osp.splitext(osp.basename(__file__))[0])

hooks = [ReconstructionVideoHook(mask_ratio=mask_ratio, fps=10, mean=_mean, std=_std)]

# Optional Weights & Biases logging (needs the wandb package). "offline"
# writes under the run dir, for `wandb sync` from a machine with internet.
# wandb = dict(project="mae_st_lcad", mode="offline")
