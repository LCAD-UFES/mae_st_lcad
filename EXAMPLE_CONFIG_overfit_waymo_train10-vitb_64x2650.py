# Overfit test: ViT-B MAE-ST from random initialization memorizing the 18 clips
# of the train10 slice (2 Waymo segments, 22 frames each) at the native range-image
# resolution, 64 beams x 2650 azimuth steps. Consumed by tools/train.py and
# tools/compute_loss.py, which copy this file verbatim into the work_dir.
#
# Hyperparameters follow the September 2026 range-lidar overfit runs. The csv is
# written on the training machine (lidar_sweep_viewer2 job prefix train10):
#     python tools/prepare_dataset.py <DATASETS_ROOT>/waymo_open_dataset/videos --prefix train10 --out <DATASETS_ROOT>/waymo_open_dataset --min-frames 20

import bootstrapper  # noqa: F401  (registers the mae_st clone)
from custom.viz_hooks import ReconstructionVideoHook

# --- What you tweak run to run ------------------------------------------------
# Warmup, then half-cycle cosine from lr to min_lr (upstream lr_sched).
lr = 1e-3
min_lr = 1e-4
warmup_epochs = 5
weight_decay = 0.0  # no regularization: the goal is to memorize
bias_wd = False
mask_ratio = 0.3
max_epochs = 3000
save_every_epochs = 50
visualize_every_iterations = 900  # every 50 epochs; one call writes 18 videos, ~35 MB
fp32 = False  # mixed precision: 8480-token sequences are memory-heavy
num_workers = 16
seed = 0
experiment_subdir = "overfit_tests"  # work_dirs/<experiment_subdir>/<this file's name>/; optional
run_tag = "baseline"  # prefixes the run and compute_loss directories; optional

# One clip per iteration, gradients accumulated over the whole slice: one optimizer
# step per epoch on the full-set gradient. accum_iter must not exceed the
# iterations per epoch (clips / batch_size), or the optimizer never steps.
batch_size = 1
accum_iter = 18
repeat_aug = 1

# --- The overfit protocol ------------------------------------------------------
# The same input every epoch: the first num_frames frames (temporal index 0), the
# whole frame (crop = frame, centre position), no flip. False restores random
# positions and flips for an ordinary training.
overfit = True
_temporal_sample_index, _spatial_sample_index, _flip = (0, 1, False) if overfit else (-1, -1, True)

_frame = (64, 2650)
_common = dict(
    path_to_data_dir="waymo_open_dataset",
    csv_file="train10.csv",  # training and validation read the same clips
    sampling_rate=1,
    num_frames=20,
    target_fps=10,
    train_crop_size=_frame,
    test_crop_size=_frame,
    train_random_horizontal_flip=_flip,
    temporal_sample_index=_temporal_sample_index,
    spatial_sample_index=_spatial_sample_index,
)
train_dataset = dict(mode="pretrain", repeat_aug=repeat_aug, **_common)
val_dataset = dict(mode="val", repeat_aug=1, **_common)

# --- Model geometry --------------------------------------------------------------
# Patch 4 beams x 50 azimuth steps, temporal patch 2: a (10, 16, 53) token grid,
# 8480 tokens. pred_t_dim=20 reconstructs every frame.
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

# panel_size=(H, W) resizes each of the four panels; None keeps the original size.
hooks = [ReconstructionVideoHook(mask_ratio=mask_ratio, panel_size=None)]

# Optional Weights & Biases logging (needs the wandb package); "offline" writes
# under the run dir, for `wandb sync` from a machine with internet.
# wandb = dict(project="mae_st_lcad", mode="offline")
