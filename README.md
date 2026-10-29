# mae_st_tooling

Tooling to train, evaluate and visualize [MAE-ST](https://github.com/facebookresearch/mae_st) (Masked Autoencoders As Spatiotemporal Learners) on lidar range-image videos, with the same workflow this group already uses for the Sapiens MAE-ViT encoder: Python configs, VSCode tasks, `work_dirs/` with a verbatim copy of the config, per-sample visualizations produced by hooks, and a loss curve. The model and data code is the `main` branch of [LCAD-UFES/mae_st_lcad](https://github.com/LCAD-UFES/mae_st_lcad): upstream MAE-ST plus the fixes and domain changes that range images need (rectangular patches and crops, no resampling, configurable sampling positions); its commit messages document each change.

Contents

1. [Why not use the mae_st scripts directly](#1-why-not-use-the-mae_st-scripts-directly)
2. [Quick start](#2-quick-start)
3. [How it is built](#3-how-it-is-built)

---

## 1. Why not use the mae_st scripts directly

`main_pretrain.py` and its siblings are single commands with ~40 argparse flags: no versionable config, no record of what ran, no validation loss, no place to plug a visualization into the loop. This directory uses the mae_st code as a library and gives it the same mode of use as Sapiens.

**Usability gaps** — things the mae_st entry points do not offer and the Sapiens workflow takes for granted:

- `main_pretrain.py` has **no validation loop**; nothing computes a loss on a second dataset during training.
- There is **no hook mechanism** in the entry points: mae_st's `train_one_epoch` takes an `after_iter` callback, and `tools/train.py` uses it to run visualizations every N iterations.
- Configuration is ~40 argparse flags; several things that matter (crop size and position, resampling, `mean`/`std` normalization, horizontal flip, `pred_t_dim` geometry) are reachable only through the Python constructors, not the CLI.
- Checkpoints are named `checkpoint-%05d.pth`, there is no "last checkpoint" sentinel, and `misc.load_model` silently auto-resumes from whatever it finds in `output_dir`.
- The published checkpoints store attention as fused `attn.qkv.*` while the model has separate `attn.q/k/v`, so `load_state_dict(strict=False)` silently leaves every attention layer at random init; `custom/checkpoints.py` splits them.

**Central goal.** Use the mae_st code as a library, unedited by the tooling, while giving it the same mode of use as the Sapiens MAE-ViT encoder — so that switching between the two encoders changes the config, not the workflow:

| Sapiens (mmengine) | here |
|---|---|
| `configs/<name>.py`, tiered, `work_dir` derived from the filename | same |
| VSCode tasks `train` / `compute_loss` / `plot_loss` | same labels, same prompts (config, tmux, checkpoint, dataset) |
| `work_dir/last_checkpoint`, `epoch_N.pth`, `<timestamp>/` per run | same layout |
| `vis_data/scalars.json` + `plot_loss.py` | same format, plus a `val_loss` curve |
| the Sapiens visualization hook (per-sample files, loss in the filename) | `custom/viz_hooks.py::ReconstructionVideoHook` |
| checkpoint every N epochs, visualize every N iterations | `save_every_epochs`, `visualize_every_iterations` (0 disables either) |

---

## 2. Quick start

### 2.0 The directories

The tooling is code only and never holds your data. The locations are all decided in one place, [`bootstrapper.py`](bootstrapper.py):

| what | default | override |
|---|---|---|
| `TOOLING_ROOT` — this directory: `custom/`, `tools/`, launchers, tasks | wherever it is checked out | — |
| `EXPERIMENTS_ROOT` — **yours**: `configs/`, `work_dirs/`, `checkpoints/` | `~/mae_st_experiments` | `$MAE_ST_EXPERIMENTS` |
| `MAE_ST_REPO_ROOT` — a clone of `mae_st_lcad`, branch `main` | `~/mae_st_lcad` | `$MAE_ST_REPO` |
| `DATASETS_ROOT` — datasets | `~/datasets` | `$MAE_ST_DATASETS` |

Every tool accepts bare names and resolves them against these roots: a config name is looked up in `EXPERIMENTS_ROOT/configs/`, a dataset name under `DATASETS_ROOT`, a checkpoint name under `EXPERIMENTS_ROOT/checkpoints/`, a `scalars.json` path under `EXPERIMENTS_ROOT/work_dirs/`. Absolute paths are always accepted as well.

### 2.1 The mae_st clone, the experiments directory, the checkpoint

Clone the `main` branch of `mae_st_lcad`. Any directory name works: the bootstrapper registers the clone as the package `mae_st`, the name its code imports itself under.

```bash
git clone https://github.com/LCAD-UFES/mae_st_lcad.git ~/mae_st_lcad
mkdir -p ~/mae_st_experiments/configs
```

Only the Kinetics example starts from a published checkpoint (~3.8 GB), placed under the experiments directory:

```bash
mkdir -p ~/mae_st_experiments/checkpoints
wget -P ~/mae_st_experiments/checkpoints https://dl.fbaipublicfiles.com/video-mae-200x4-nonorm.pth
```

### 2.2 Environment

Building the Python environment for `mae_st` is its own trial-and-error session (a bundled `environment.yml` that does not work, dependencies frozen at old releases, torch builds that differ by CUDA runtime). [`_install/`](_install/README.md) makes it reproducible: an installer driven by a per-platform manifest recording what to install and why, which ends by running the test suites of both clones. Follow its README once per machine; [`_install/DESIGN.md`](_install/DESIGN.md) records the reasoning behind each pin.

Every command below assumes the resulting environment is active. The launchers (`run_training.sh`, `run_tensorboard.sh`, `run_tool.sh`, hence the VSCode tasks) activate it through `env.sh`, which finds conda from `CONDA_EXE` or `PATH` and activates `$MAE_ST_CONDA_ENV`, default `mae_st`.

### 2.3 A dataset (the minimum the loader needs)

The `Kinetics` dataset class needs a directory holding `train.csv`, `val.csv` and `test.csv`. Each line is `absolute_path_to_video label` (one space, label an integer, ignored in pretraining but it must parse). Paths cannot contain spaces. Which csv is read is decided by the dataset *mode*: `pretrain`→`train.csv`, `val`→`val.csv`, `test`→`test.csv`.

Range-image clips come from [lidar_sweep_viewer2](https://github.com/LCAD-UFES/lidar_sweep_viewer2), which writes a folder of mp4s and no csvs. `tools/prepare_dataset.py` writes them, on the machine that trains, since the paths are absolute:

```bash
python tools/prepare_dataset.py ~/datasets/range_lidar_slice10 --min-frames 20   # every clip in all three csvs (overfit protocol)
python tools/prepare_dataset.py ~/datasets/my_split --split prefix                # train_/val_/test_ file prefixes
```

`--min-frames` drops clips shorter than the config's `num_frames`; `--limit N` keeps the first N per split.

A 10-video slice of Kinetics-400 is built by `tools/prepare_k400_slice.py` into `~/datasets/kinetics400_slice10/` (10 classes, short edge 256, `labels.txt` with the alphabetical class ids). It downloads only the first ~150 MB of one official tarball, validates each video by decoding it, resizes with PyAV (no `ffmpeg` CLI needed) and writes the csvs — see its `--help` for a larger slice.

### 2.4 An elementary config

A config is a Python file in `EXPERIMENTS_ROOT/configs/`; the drivers execute it and read its top-level names. Because the drivers have already put the tooling on `sys.path`, a config needs no path logic at all: it imports `bootstrapper` for locations and `custom.*` for hooks.

Two complete, commented ones ship here — copy one into `EXPERIMENTS_ROOT/configs/`, drop the `EXAMPLE_CONFIG_` prefix and edit from there (the `work_dir` follows the file name, so renaming it names the experiment); the configs you actually run live in your experiments directory:

- [`EXAMPLE_CONFIG_overfit_range_lidar_slice10-vitb_64x2650.py`](EXAMPLE_CONFIG_overfit_range_lidar_slice10-vitb_64x2650.py): ViT-B from scratch overfitting range-image clips at their native 64 x 2650 (patch 4 x 50, 8480 tokens), the same window every epoch.
- [`EXAMPLE_CONFIG_overfit_k400_slice10-vitl_nonorm.py`](EXAMPLE_CONFIG_overfit_k400_slice10-vitl_nonorm.py): ViT-L from the published checkpoint on Kinetics, with upstream's scale jitter (`spatial_resample=True`).

This is a Kinetics config cut down to the minimum:

```python
# ~/mae_st_experiments/configs/my_first_test.py
import os.path as osp
import bootstrapper
from custom.viz_hooks import ReconstructionVideoHook

lr = min_lr = 1e-5; warmup_epochs = 0        # constant LR (see §3.7)
mask_ratio = 0.9
max_epochs = 20
save_every_epochs = 5                        # 0 = only the final checkpoint
visualize_every_iterations = 25              # 0 = never during training
batch_size = 2; repeat_aug = 2; seed = 0

_common = dict(path_to_data_dir="kinetics400_slice10",   # a name under DATASETS_ROOT
               sampling_rate=4, num_frames=16, spatial_resample=True)
train_dataset = dict(mode="pretrain", repeat_aug=repeat_aug, **_common)
val_dataset   = dict(mode="val", repeat_aug=1, train_random_horizontal_flip=False, **_common)

model_name = "mae_vit_large_patch16"
model_kwargs = dict(num_frames=16, t_patch_size=2, decoder_embed_dim=512, decoder_depth=4,
                    pred_t_dim=16, norm_pix_loss=False)      # geometry of the checkpoint below
load_from = "video-mae-200x4-nonorm.pth"                     # a name under EXPERIMENTS_ROOT/checkpoints/

work_dir = bootstrapper.work_dir_for(osp.splitext(osp.basename(__file__))[0])
hooks = [ReconstructionVideoHook(mask_ratio=mask_ratio)]
```

### 2.5 Run it

Open this directory in VSCode (alone, or as one folder of a multi-root workspace together with the experiments directory and the clone). The tasks live in `.vscode/tasks.json` here (`Terminal → Run Task…`, or the TASKS panel):

| task | asks for | does |
|---|---|---|
| `train` | config, tmux?, checkpoint | trains; writes `EXPERIMENTS_ROOT/work_dirs/mae_st_overfit_tests/<config>/` |
| `compute_loss` | config, checkpoint, dataset | loss per video + visualizations for one checkpoint |
| `plot_loss` | `scalars.json` path | PNG with train and val loss curves |
| `tensorboard` | — | dashboard over *all* work_dirs, in the foreground (stop with Ctrl+C) |

The tasks are only a front end: each one answers its prompts and runs a command you can type yourself, from any directory, with the environment active. The table below is exhaustive — every path the prompts can take, plus the options only the command line exposes. `~/mae_st_tooling` is wherever this directory lives; a config, dataset or checkpoint may be given as an absolute path or as a bare name (looked up in `EXPERIMENTS_ROOT/configs/`, `DATASETS_ROOT`, `EXPERIMENTS_ROOT/checkpoints/` respectively).

| task | prompt / answer | command |
|---|---|---|
| `train` | config: any name or path | `python ~/mae_st_tooling/tools/train.py my_config.py` |
| `train` | checkpoint: **blank** — start from scratch (weights from the config's `load_from`, or random if it has none) | `… tools/train.py my_config.py` |
| `train` | checkpoint: **`last`** — resume the newest checkpoint of this config | `… tools/train.py my_config.py --checkpoint last` |
| `train` | checkpoint: **a path** — resume that `.pth` | `… tools/train.py my_config.py --checkpoint <work_dir>/epoch_20.pth` |
| `train` | tmux: **Não** — run in this terminal | any of the commands above, or `bash ~/mae_st_tooling/run_training.sh my_config.py Não '<checkpoint>'` |
| `train` | tmux: **Sim** — detached, survives disconnection | `bash ~/mae_st_tooling/run_training.sh my_config.py Sim '<checkpoint>'` then `tmux attach -t <config>-<timestamp>` |
| `train` | *(not in the task)* do not write TensorBoard events | `… tools/train.py my_config.py --tensorboard off` |
| `compute_loss` | checkpoint: **`last`** (the task's default) — newest checkpoint of this config | `python ~/mae_st_tooling/tools/compute_loss.py my_config.py --checkpoint last --dataset my_dataset` |
| `compute_loss` | checkpoint: **blank** — the config's `load_from`, i.e. the point training starts from | `… tools/compute_loss.py my_config.py --dataset my_dataset` |
| `compute_loss` | checkpoint: **a path or name** — including a published, fused-qkv one | `… tools/compute_loss.py my_config.py --checkpoint video-mae-200x4-nonorm.pth --dataset my_dataset` |
| `compute_loss` | dataset: any dir holding the csvs | `… --dataset ~/datasets/kinetics400_slice10` |
| `compute_loss` | *(not in the task)* read `train.csv` or `test.csv` instead of `val.csv` | `… --dataset my_dataset --mode pretrain` (or `--mode test`) |
| `compute_loss` | *(not in the task)* override the batch size | `… --dataset my_dataset --batch-size 4` |
| `plot_loss` | log_file: a `scalars.json` | `python ~/mae_st_tooling/tools/plot_loss.py <run>/vis_data/scalars.json` |
| `plot_loss` | log_file: a run directory | `… tools/plot_loss.py mae_st_overfit_tests/my_config/<timestamp>` |
| `plot_loss` | log_file: a config's work_dir (the task's default) — plots its newest run | `… tools/plot_loss.py mae_st_overfit_tests/my_config/` |
| `plot_loss` | *(not in the task)* choose where the PNG goes | `… tools/plot_loss.py <scalars.json> --output /tmp/curve.png` |
| `tensorboard` | — | `bash ~/mae_st_tooling/run_tensorboard.sh` |
| `tensorboard` | *(not in the task)* a different logdir or port | `bash ~/mae_st_tooling/run_tensorboard.sh <logdir> <port>` |

Two things the command line does not change: `train.py` always writes checkpoints as the config's `save_every_epochs` dictates, and a resume always opens a *new* `<timestamp>/` run directory while carrying the previous `scalars.json` records forward, so the loss curve stays continuous.

What a training leaves behind, under `EXPERIMENTS_ROOT`:

```
work_dirs/mae_st_overfit_tests/my_first_test/
├── my_first_test.py            verbatim copy of the config used
├── last_checkpoint             one line: absolute path of the newest epoch_N.pth
├── epoch_15.pth, epoch_20.pth  N = epochs completed; pruned to max_keep_ckpts (2)
├── <timestamp>/                one per train.py invocation (a resume opens a new one)
│   ├── <timestamp>.log         everything printed to the console
│   ├── my_first_test.py        the config again
│   ├── tb/                     TensorBoard events (train_loss/lr per iteration, val_loss per epoch)
│   └── vis_data/
│       ├── scalars.json        one JSON per line: {epoch, loss, val_loss, lr, time}
│       └── epoch0005_iter000025/<video>_loss0.0823.mp4   original, masked, reconstruction, error
└── compute_loss/<timestamp>-epoch_20/
    ├── <timestamp>.log, <timestamp>.json   mean loss + per-sample losses
    └── vis_data/eval/<video>_loss0.0801.mp4
```

Checkpoints are ~3.8 GB each (ViT-L plus optimizer state) — mind `save_every_epochs` and `max_keep_ckpts` on long runs.

---

## 3. How it is built

### 3.1 Code map

```
   <MAE_ST_REPO_ROOT>  (mae_st_lcad main, imported as package mae_st.*)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ models_mae.py   util/kinetics.py   util/misc.py   util/lr_sched.py   │
   │ engine_pretrain.py (train_one_epoch, after_iter)   util/decoder/     │
   └──────▲───────────────▲───────────────▲───────────────▲───────────────┘
          │ build model    │ build dataset │ AdamW groups, │ lr per iter,
          │                │               │ NativeScaler  │ train step
 <mae_st_tooling>  (code)  │               │               │
 ┌────────┼────────────────┼───────────────┼───────────────┼─────────────┐
 │ bootstrapper.py ── the setup: locations (TOOLING/EXPERIMENTS/MAE_ST_   │
 │                    REPO/DATASETS roots), sys.path, the mae_st package; │
 │                    imported first by every driver and by every config  │
 │ custom/experiment.py ── config loading, run dirs, console tee, fixed   │
 │                     val clips, deterministic val loss, hooks runner,   │
 │                     checkpoint save/resume/sentinel, scalars.json      │
 │ custom/checkpoints.py ── loads published checkpoints (fused qkv split) │
 │ custom/viz_hooks.py ── ReconstructionVideoHook (per-sample mp4)        │
 │ tools/train.py, compute_loss.py, plot_loss.py, prepare_dataset.py      │
 │ env.sh (conda), run_training.sh (tmux, resume), run_tool.sh,           │
 │ run_tensorboard.sh, .vscode/tasks.json, tools/prepare_k400_slice.py    │
 │ EXAMPLE_CONFIG_*.py ── templates to copy into EXPERIMENTS_ROOT/configs/│
 │ tests/ ── unittest, CPU                                                │
 └────────┬───────────────────────────────────────────────────────────────┘
          │ reads configs/, writes work_dirs/, loads checkpoints/
 <EXPERIMENTS_ROOT>  (yours; default ~/mae_st_experiments)
 ┌────────▼───────────────────────────────────────────────────────────────┐
 │ configs/<name>.py ──► work_dirs/mae_st_overfit_tests/<name>/           │
 │ checkpoints/*.pth                                                      │
 └────────────────────────────────────────────────────────────────────────┘
```

The one rule: `bootstrapper` must be imported before anything imports `mae_st`. `custom/experiment.py` imports it at the top, which is why every driver starts with `from custom import experiment` and only then imports from `mae_st`.

### 3.2 `bootstrapper.py`

The whole initial setup, as a side effect of `import bootstrapper`: it computes the four roots (this directory; `EXPERIMENTS_ROOT`, `MAE_ST_REPO_ROOT`, `DATASETS_ROOT` from their defaults or the `MAE_ST_*` environment variables), puts the tooling on `sys.path` and registers the mae_st clone as the package `mae_st`, whatever its directory name; a missing clone fails at import, naming the variable to set. It also provides `work_dir_for(config_name)` (`EXPERIMENTS_ROOT/work_dirs/mae_st_overfit_tests/<name>`) and `resolve(path, base)`, the bare-name resolution every tool and prompt relies on. `python bootstrapper.py` prints the roots in effect.

### 3.3 `custom/experiment.py`

Everything `train.py` and `compute_loss.py` share.

`load_config(name_or_path)` resolves the config (bare names in `EXPERIMENTS_ROOT/configs/`), runs it with `runpy` and returns its namespace as a dict (required keys are checked). `make_run_dir()` creates `work_dir/<ts>/vis_data/` and copies the config verbatim to `work_dir/` and `work_dir/<ts>/`. `install_tee()` mirrors stdout/stderr into `<ts>.log` — upstream prints go through the builtin `print`, so its progress lines are captured too. `build_dataset()` resolves `path_to_data_dir` against `DATASETS_ROOT`.

`load_fixed_clips(dataset, n, seed)` decodes the validation clips **once**, under a fixed seed, and keeps them in memory. Three RNGs feed a Kinetics sample (Python `random` for the temporal start, `numpy` for crops/flips, torch for the MAE mask); re-seeding them every epoch would also reset the training sampler's shuffle. Freezing the clips and forking only the torch RNG around each evaluation (`fork_seeded`) makes the validation masks identical from epoch to epoch without touching training randomness. That is what makes `val_loss` comparable over time, and why `compute_loss` on `epoch_N.pth` reproduces exactly the `val_loss` train.py logged. `masked_loss()` is that evaluation; `run_hooks()` runs the hooks under the same envelope and merges their `{stem: loss}` results.

Checkpoints: `save_checkpoint()` writes `epoch_N.pth` with the upstream dict keys (`model`, `optimizer`, `epoch`, `scaler`, `args`) plus `config` and `run_dir`, rewrites `last_checkpoint`, prunes to `max_keep_ckpts`. `resolve_checkpoint(work_dir, explicit, load_from)` is the one place that answers "which weights": `last` → the sentinel (error if absent), an explicit path or name → itself, nothing → the config's `load_from`. It returns the path *and* which of those cases it was, which is how `train.py` knows whether to resume or start fresh and what to print. `load_resume_checkpoint()` restores model/optimizer/scaler, returns the next epoch to run, and refuses checkpoints this tooling did not write.

### 3.4 The training step

`tools/train.py` runs mae_st's `engine_pretrain.train_one_epoch`, which calls `after_iter(global_iter)` at the end of every step (1-based across epochs); that callback is where the hooks fire every `visualize_every_iterations`.

One subtlety it carries: on a non-finite loss the engine deletes `num_checkpoint_del` files chosen by `misc.get_last_checkpoint`, whose filter is `"checkpoint" in filename` — which matches our `last_checkpoint` sentinel. `train.py` therefore passes `num_checkpoint_del=0`; the engine then only raises.

### 3.5 `custom/checkpoints.py`

`load_mae_checkpoint(model, path)`: loads a pretrain checkpoint into a `models_mae` model, converting the fused `attn.qkv.{weight,bias}` of the published checkpoints into the separate `attn.q/k/v` that upstream `video_vit.Attention` now uses (a 3-way chunk along dim 0, the timm layout), for encoder and decoder blocks alike. It raises if any model key is left uninitialized — the failure mode it prevents is a model that loads "successfully" with random attention weights. Our own `epoch_N.pth` files already have split keys and pass through unchanged.

### 3.6 `custom/viz_hooks.py`

The hook contract: a callable `hook(model, samples, stems, tag, run_dir) -> {stem: loss}` where `samples` is a normalized `[B, 3, T, H, W]` batch on the model's device, `stems` names each sample, `tag` labels the moment (`epoch0005_iter000025`, `eval`) and outputs go under `run_dir/vis_data/<tag>/`. Hooks are instantiated **in the config**, so a new visualization is a new callable listed there — no driver changes.

`ReconstructionVideoHook` writes one mp4 per sample with four panels at the model's input resolution: original, masked input (masked patches in gray), reconstruction, and the per-pixel error. With `layout="auto"` (default) they are stacked top to bottom for frames wider than tall — a 64 x 2650 range image gives a 256 x 2650 video — and side by side otherwise; `"vertical"`/`"horizontal"` force one. Its `mean`/`std` must be the dataset's normalization. The reconstruction panel is the *composite* the MAE papers show — true pixels at visible patches, prediction only at masked ones — because the loss only ever penalizes masked patches and the decoder's output at visible positions is untrained noise. The filename carries the per-sample masked-region loss (`<stem>_loss0.0823.mp4`), computed with the same formula as `forward_loss` reduced per sample. `norm_pix_loss` checkpoints are handled (prediction is de-normalized with the target's per-patch statistics before rendering).

### 3.7 `tools/train.py`

```
python tools/train.py <config> [--checkpoint last|PATH] [--tensorboard on|off]
```

Owns the epoch loop; mae_st supplies the parts. Per config it builds the `Kinetics` train dataset (mode `pretrain`) and a plain `DataLoader` (`drop_last=False` — upstream's `True` would drop videos from a 10-video set), the fixed validation clips, the model (initialized from `load_from` when not resuming), `misc.add_weight_decay` parameter groups, `torch.optim.AdamW(betas=(0.9, 0.95))` and `misc.NativeScalerWithGradNormCount`. It assembles the `argparse.Namespace` the upstream engine and `lr_sched` read (`accum_iter`, `mask_ratio`, `clip_grad`, `repeat_aug`, `lr`, `min_lr`, `warmup_epochs`, `epochs`, `output_dir`, `num_checkpoint_del=0`).

Each epoch: `custom.engine.train_one_epoch` (with the hooks firing from `after_iter` when `visualize_every_iterations > 0`), then `val_loss` on the fixed clips, a line appended to `scalars.json`, and a checkpoint when `save_every_epochs` divides the epoch count (always on the last epoch). Epoch numbers shown to the user are *epochs completed*, 1-based; the engine internally receives the upstream 0-based index.

**Learning rate.** Upstream's `lr_sched` applies linear warmup then half-cycle cosine from `lr` to `min_lr`, unconditionally. For a constant rate set `lr == min_lr` and `warmup_epochs = 0`. `lr` may be left `None` to use `blr` with upstream's linear scaling rule (`blr × effective batch / 256`).

**Where it starts.** One argument decides, and the run prints which case it took. With no `--checkpoint` it is a fresh run: epoch 1, weights from the config's `load_from` (split from fused qkv when needed) or random initialization if there is none. With `--checkpoint last` it resumes from the newest checkpoint of that config, named by `work_dir/last_checkpoint`; with `--checkpoint PATH` (or a bare name under `checkpoints/`) it resumes from that file. Resuming restores model, optimizer, scaler and epoch counter, and opens a *new* `<ts>/` run directory (own log, config copy, tb) pre-filled with the previous run's `scalars.json` records, so the curve stays continuous.

Nothing here is guessed. `--checkpoint last` without a sentinel is an error, not a silent fresh start; and a checkpoint that this tooling did not write — one with no optimizer state, or with fused `attn.qkv` weights, like the published ones — is refused in one line telling you to put it in `load_from` instead, because its optimizer state cannot match a model whose q/k/v are separate.

**Gradient accumulation.** The engine steps the optimizer every `accum_iter` iterations, counting from 0 each epoch. An `accum_iter` larger than the iterations per epoch (clips / `batch_size`) would never step, so `train.py` refuses it; one that does not divide them is a warning, since the leftover gradients carry into the next epoch's first step.

**Weights & Biases.** Optional: a config that defines `wandb = dict(project=..., mode="offline")` gets one wandb run per invocation with the per-epoch `loss`, `val_loss` and `lr`; `mode="offline"` writes under the run dir for a later `wandb sync`. Configs without it do not need the package.

**TensorBoard.** Events go to `<run>/tb` by default (`flush_secs=10`), which is what allows the dashboard to be attached later; `--tensorboard off` disables the writer (`scalars.json` is always written).

### 3.8 `tools/compute_loss.py`

```
python tools/compute_loss.py <config> --dataset DIR [--checkpoint PATH] [--mode val|pretrain|test]
```

Inference over a dataset directory with one checkpoint, chosen by the same scheme as `train.py`: `--checkpoint last` is the newest checkpoint of this config, an explicit path or name (a published fused-qkv one included) is used as given, and omitting it evaluates the config's `load_from` — the weights a fresh training would start from, which makes "before training" a one-command baseline. Uses the config's `val_dataset` settings with `path_to_data_dir` replaced by `--dataset` and the csv chosen by `--mode`. Runs the config's hooks (or a default `ReconstructionVideoHook`) over every clip with the same fixed masking as training's `val_loss`, prints the per-sample table and the mean, and writes `work_dir/compute_loss/<ts>-<checkpoint_stem>/` with `<ts>.log`, `<ts>.json` and `vis_data/eval/`.

Because the masks and clips are fixed, the number it reports is a property of the checkpoint and the dataset only, not of the run — so it can be compared across checkpoints, and against the `val_loss` column of `scalars.json`. Two uses follow from that: pointing `--dataset` at the training set gives the training loss measured *without* the training-time augmentation randomness (the per-iteration `loss` in `scalars.json` is averaged over random clips and masks, so it is noisier and not directly comparable); pointing it at a dataset the model never saw gives the loss on unseen data, and the difference between the two is the generalization gap of that checkpoint.

### 3.9 `tools/plot_loss.py`

```
python tools/plot_loss.py <scalars.json> [--output PATH]
```

Reads the JSON-lines file (a path, or one relative to `EXPERIMENTS_ROOT/work_dirs/`; a run directory or a config's work_dir is accepted too, the latter plotting its newest run — which is what the task's default prompt value does), plots `loss` and (when present) `val_loss` against epoch, saves `<input>_loss_curve.png` next to the input. Malformed lines are skipped with a warning.

### 3.10 `run_training.sh`, `run_tensorboard.sh`, `.vscode/`

`env.sh` is sourced by every launcher: it finds conda (`CONDA_EXE`, else `conda info --base`) and activates `$MAE_ST_CONDA_ENV` (default `mae_st`). `run_tool.sh <name> [args]` runs `tools/<name>.py` inside it and backs the `compute_loss` and `plot_loss` tasks.

`run_training.sh <config> <Sim|Não> [checkpoint]` is the launcher behind the `train` task, same design as the Sapiens one: every decision arrives as an argument (answered by VSCode's prompts before the script runs, so nothing ever blocks on a `read` inside a possibly flaky SSH session); it asks the tooling to resolve the config and its `work_dir`, echoes both, passes the third argument straight through to `--checkpoint`, writes the inner command to a temp script and runs it either inside a detached tmux session named `<config>-<timestamp>` (survives connection drops; `tmux attach -t …`) or in the foreground. It decides nothing about checkpoints itself — `train.py` resolves and reports that — and holds no paths of its own.

`run_tensorboard.sh [logdir] [port]` runs TensorBoard in the *foreground* over all of `EXPERIMENTS_ROOT/work_dirs/` (so runs from every config, finished or in progress, can be overlaid), bound to localhost only — TensorBoard has no authentication. Started from the `tensorboard` task it is stoppable from the task panel, and VSCode's Remote-SSH detects the port and offers to open it (`.vscode/settings.json` labels port 6006). Independent of the editor, a tunnel from your machine reaches the same server: `autossh -M 0 -N -L 6006:localhost:6006 <user>@<host>`.

`.vscode/tasks.json` defines the four tasks. In a multi-root workspace `${workspaceFolder}` cannot be resolved, so the commands spell the tooling location as `~/mae_st_tooling`, the one thing to update there if it lives elsewhere. Labels intentionally carry no `.py`.

### 3.11 `tools/prepare_dataset.py` and `tools/prepare_k400_slice.py`

`prepare_dataset.py <videos_dir> [--split all|prefix] [--limit N] [--min-frames K] [--out DIR]` writes the three csvs for a folder of mp4s (§2.3).

`prepare_k400_slice.py` builds the Kinetics slice: annotations and a partial download of one validation tarball from the CVD Foundation S3 mirror (tar+gzip is a stream, so the first 150 MB already hold dozens of complete videos), a full-decode check to drop the truncated tail, selection of N videos of distinct classes, PyAV resize to short edge 256 with libx264, alphabetical class ids in `labels.txt`, and the same videos written to all three csvs (`--out-dir`, `--raw-dir`, `--n-videos`, `--partial-mb`).

### 3.12 Tests

`python -m unittest discover -s tests -t .` from this directory, with `$MAE_ST_REPO` pointing at the mae_st clone. They run on CPU: the hook layouts, and an end-to-end run over two synthetic videos (prepare_dataset, two epochs of train.py with a stand-in wandb, compute_loss on the last checkpoint, the accum_iter refusal).
