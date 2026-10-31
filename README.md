# mae_st_lcad

MAE-ST ([Masked Autoencoders As Spatiotemporal Learners](https://arxiv.org/abs/2205.09113), Feichtenhofer et al., 2022) adapted to lidar range-image videos.

This repository is a fork of [facebookresearch/mae_st](https://github.com/facebookresearch/mae_st). Upstream trains a video encoder by masking most of a clip and reconstructing it, on natural-image video (Kinetics). Here the same model learns from the range images of a rotating lidar: each frame is one sweep, a row is a laser beam and a column an azimuth step (64 x 2650 for the Waymo top lidar, 10 frames per second).

That change of domain is not just a different dataset. Range images are a few rows tall and thousands of columns wide, so frames and patches are not square; and their pixels are measurements, so the image must never be interpolated. Branch `main` holds upstream's code plus the commits that make this work: rectangular patches and crops, no resampling by default, fixed clip and crop positions for overfit tests, a PyAV video decoder, and the fixes needed to run on a current PyTorch outside Meta's infrastructure. [DESIGN.md](DESIGN.md) explains each change and why it was made.

The upstream documentation (fine-tuning, pre-training on Kinetics, the published checkpoints) is kept as it was: [PRETRAIN.md](PRETRAIN.md), [FINETUNE.md](FINETUNE.md), [DATASET.md](DATASET.md), and upstream's README at [facebookresearch/mae_st](https://github.com/facebookresearch/mae_st/blob/main/README.md).

## How to use it

`main` is used as a library, not run directly. Training, evaluation and visualization go through the tooling in branch [`mae_st_tooling`](https://github.com/LCAD-UFES/mae_st_lcad/tree/mae_st_tooling): a training is described by a Python config file, launched by one command (optionally, made more convenient with a VSCode task), and leaves behind a directory with the config it used, checkpoints, the loss curve and reconstruction videos.

The work is spread over four directories, each in its own place, so code, experiments and data never mix:

| directory | what | default |
|---|---|---|
| this repository, branch `main` | the model and data code | `~/mae_st_lcad` |
| this repository, branch `mae_st_tooling` | the tooling and the environment installer | `~/mae_st_tooling` |
| experiments | your configs, and what trainings write | `~/mae_st_experiments` |
| datasets | videos and their csv lists | `~/datasets` |

Each default can be changed with an environment variable (`MAE_ST_REPO`, `MAE_ST_EXPERIMENTS`, `MAE_ST_DATASETS`).

### Minimal training

```bash
# 1. code
git clone https://github.com/LCAD-UFES/mae_st_lcad.git ~/mae_st_lcad
git clone -b mae_st_tooling https://github.com/LCAD-UFES/mae_st_lcad.git ~/mae_st_tooling

# 2. environment (conda), once per machine; see _install/README.md for a machine without a manifest
cd ~/mae_st_tooling/_install && ./conda.sh deps.linux-aarch64-cu128.tsv
conda activate mae_st

# 3. dataset: videos from lidar_sweep_viewer2, then the csv that lists them
python ~/mae_st_tooling/tools/prepare_dataset.py ~/datasets/waymo_open_dataset/videos \
    --prefix train10 --out ~/datasets/waymo_open_dataset --min-frames 20

# 4. config
mkdir -p ~/mae_st_experiments/configs
cp ~/mae_st_tooling/EXAMPLE_CONFIG_overfit_waymo_train10-vitb_64x2650.py \
   ~/mae_st_experiments/configs/my_first_run.py

# 5. train (in tmux, so it survives a dropped connection), then plot the loss
bash ~/mae_st_tooling/run_training.sh my_first_run.py Sim
python ~/mae_st_tooling/tools/plot_loss.py mae_st_overfit_tests/my_first_run/
```

The videos come from [lidar_sweep_viewer2](https://github.com/LCAD-UFES/lidar_sweep_viewer2), which converts Waymo Open Dataset segments into range-image clips. The tooling's [README](https://github.com/LCAD-UFES/mae_st_lcad/blob/mae_st_tooling/README.md) covers every command, option and output.

## Branches

- `main`: the model and data code.
- `mae_st_tooling`: the tooling.

Earlier states of the project are kept as tags `archive/*`.

## License

CC-BY-NC 4.0, as upstream: see [LICENSE](LICENSE).
