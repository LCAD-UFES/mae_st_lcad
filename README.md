# mae_st_tooling

Train, evaluate and visualize the MAE-ST model of [mae_st_lcad](https://github.com/LCAD-UFES/mae_st_lcad) `main` on lidar range-image videos. A training is a Python config file; one command (or VSCode task) runs it and leaves a directory with the config it used, checkpoints, the loss curves and reconstruction videos. The tooling uses `main` as a library and never edits it. [CONTRIBUTING.md](CONTRIBUTING.md) explains how it is built.

## 1. Setup

### Directories

The tooling holds code only. [`bootstrapper.py`](bootstrapper.py) decides where everything else is:

| root | holds | default | override |
|---|---|---|---|
| `MAE_ST_REPO_ROOT` | a clone of `mae_st_lcad`, branch `main` | `~/mae_st_lcad` | `$MAE_ST_REPO` |
| `EXPERIMENTS_ROOT` | yours: `configs/`, `work_dirs/`, `checkpoints/` | `~/mae_st_experiments` | `$MAE_ST_EXPERIMENTS` |
| `DATASETS_ROOT` | datasets | `~/datasets` | `$MAE_ST_DATASETS` |

Every tool accepts bare names and looks them up there: a config in `EXPERIMENTS_ROOT/configs/`, a dataset under `DATASETS_ROOT`, a checkpoint in `EXPERIMENTS_ROOT/checkpoints/`, a run under `EXPERIMENTS_ROOT/work_dirs/`. `python bootstrapper.py` prints the roots in effect.

```bash
git clone https://github.com/LCAD-UFES/mae_st_lcad.git ~/mae_st_lcad
git clone -b mae_st_tooling https://github.com/LCAD-UFES/mae_st_lcad.git ~/mae_st_tooling
mkdir -p ~/mae_st_experiments/configs
```

### Environment

[`_install/`](_install/README.md) builds the conda environment `mae_st` from a per-platform manifest and finishes by running both test suites. The launchers (`run_training.sh`, `run_tool.sh`, `run_tensorboard.sh`, hence the VSCode tasks) activate it on their own through `env.sh`; for the `python tools/...` commands below, `conda activate mae_st` first.

### Dataset

The loader reads a directory with csv files, one `absolute_path_to_video label` per line (the label is ignored in pretraining). Range-image clips come from [lidar_sweep_viewer2](https://github.com/LCAD-UFES/lidar_sweep_viewer2), which writes mp4s and no csvs; `tools/prepare_dataset.py` writes them on the machine that trains, since the paths are absolute:

```bash
# one csv per lsv2 slice (the job's `prefix`), next to the videos folder: train10_*.mp4 -> train10.csv
python tools/prepare_dataset.py ~/datasets/waymo_open_dataset/videos --prefix train10 --out ~/datasets/waymo_open_dataset --min-frames 20

# a folder that is one slice: all its videos in train.csv, val.csv and test.csv (overfit protocol)
python tools/prepare_dataset.py ~/datasets/my_clips --min-frames 20
```

A config reads a named csv with `csv_file="train10.csv"` in its dataset; without it the loader reads `train.csv`, `val.csv` or `test.csv` by mode. `--min-frames` drops clips shorter than the config's `num_frames`; `--help` lists the other options.

### Config

A config is a Python file in `EXPERIMENTS_ROOT/configs/`; its file name names the experiment. Copy the example and edit from there:

- [`EXAMPLE_CONFIG_overfit_waymo_train10-vitb_64x2650.py`](EXAMPLE_CONFIG_overfit_waymo_train10-vitb_64x2650.py): ViT-B from scratch memorizing the 18 clips of `train10` at the native 64 x 2650 resolution (patch 4 x 50, 8480 tokens), the same window every epoch.

## 2. Run

From VSCode (`Terminal → Run Task…`), or the command each task runs:

| task | command | does |
|---|---|---|
| `train` | `bash run_training.sh <config> Sim\|Não [checkpoint]` | trains, in tmux (`Sim`) or in the terminal; checkpoint blank = fresh, `last` = resume |
| `compute_loss` | `python tools/compute_loss.py <config> --dataset <dir> --checkpoint last` | loss per clip and reconstruction videos for one checkpoint |
| `plot_loss` | `python tools/plot_loss.py <run or work_dir>` | PNG with the train and val loss curves |
| `tensorboard` | `bash run_tensorboard.sh` | dashboard over every run, on localhost:6006 |

`python tools/<tool>.py --help` documents every option, including those the tasks do not ask for (`compute_loss --mode`, `--csv`, `--batch-size`; `train --tensorboard off`). A training in tmux is reached with `tmux attach -t <config>-<timestamp>` (detach: `Ctrl+B`, `D`).

### What a training leaves

```
EXPERIMENTS_ROOT/work_dirs/[<experiment_subdir>/]<config>/
├── <config>.py                 the config used
├── last_checkpoint             path of the newest epoch_N.pth
├── epoch_N.pth                 every save_every_epochs; the newest max_keep_ckpts (2) are kept
├── [<run_tag>-]<timestamp>/    one per invocation; a resume opens a new one and continues the curve
│   ├── <timestamp>.log         the console
│   ├── tb/                     TensorBoard events
│   └── vis_data/
│       ├── scalars.json        one line per epoch: {epoch, loss, val_loss, lr, time}
│       └── epoch*_iter*/<clip>_loss<value>.mp4   panels: original, masked, reconstruction, error
└── compute_loss/[<run_tag>-]<timestamp>-<checkpoint>/   log, json with the losses, videos
```

The config's file name names this directory; an optional `experiment_subdir = "overfit_tests"` (nested paths allowed) groups it with others under `work_dirs/`. An optional `run_tag = "lr1e-3"` prefixes the run and compute_loss directories (and the tmux session) so a group of runs is easy to find. It labels runs, not checkpoints: `epoch_N.pth` and `last_checkpoint` belong to the config, so a variant to compare against goes in a config with its own name.

### Resources

Measured 2026-11-06 on a DGX Spark (GB10) with the example config: 0.5 s per iteration at batch 1, 5 GB peak memory, ~18 s per epoch of 18 clips including validation (which runs in fp32 and takes about half of it), so ~15 h for 3000 epochs. `attn_impl="explicit"` in `model_kwargs` restores the attention of runs made before 2026-11-05, bit for bit and at their cost: 3.7 s per iteration, 68 GB, ~74 h. A checkpoint is 1.3 GB; a visualization call writes ~30 MB and takes ~15 s, so keep `visualize_every_iterations` in the hundreds on long runs. Two runs of one config do not give the same curve (see `DESIGN.md` §6 of `main`), so compare configurations over more than one run. `ReconstructionVideoHook(panel_size=(H, W))` resizes each of the four panels, larger to inspect or smaller to save disk.

## 3. Tests

```bash
python -m unittest discover -s tests -t .
```

CPU, a few seconds: the visualization layouts, `prepare_dataset.py`, and a two-epoch training through `tools/train.py` followed by `compute_loss`.
