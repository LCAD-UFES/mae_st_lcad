# mae_st_tooling

Tooling to train, evaluate and visualize [MAE-ST](https://github.com/facebookresearch/mae_st) (Masked Autoencoders As Spatiotemporal Learners) from a pristine, unmodified clone of the upstream repository, with the same workflow this group already uses for the Sapiens MAE-ViT encoder: Python configs, VSCode tasks, `work_dirs/` with a verbatim copy of the config, per-sample visualizations produced by hooks, and a loss curve.

Contents

1. [Why not use the mae_st scripts directly](#1-why-not-use-the-mae_st-scripts-directly)
2. [Quick start](#2-quick-start)
3. [How it is built](#3-how-it-is-built)

---

## 1. Why not use the mae_st scripts directly

`facebookresearch/mae_st` was published straight out of Meta's internal monorepo and assumes Meta's software stack: on any other stack its entry points stop at a handful of specific imports and calls, listed below. Once past those, each script is a single command with ~40 argparse flags — no versionable config, no record of what ran, no second-dataset loss, no place to plug a visualization into the loop. This directory fixes the first problem with shims applied from outside the clone, and solves the second by giving MAE-ST the same mode of use as Sapiens.

**Stack incompatibilities** (each one is an import or call that fails outright):

| where | what | effect |
|---|---|---|
| `util/misc.py:23` | `import torch.fb.rendezvous.zeus` — a Meta-internal module, present since the first commit, never used | nothing that imports `misc` loads |
| `util/decoder/decoder.py` | video decoding goes through torchvision's *private* in-memory API (`io._probe_video_from_memory`, `io._read_video_from_memory`), since removed from torchvision | every clip fails to decode; the failure is swallowed (`return None`) and surfaces far away as `TypeError: cannot unpack non-iterable NoneType` in `kinetics.py` |
| `util/meters.py:7` | `from sklearn.metrics import ...` — an extra dependency used only by the multi-label mAP path | `main_test` does not import unless scikit-learn is installed |
| `main_pretrain.py:298` | `torch.optim._multi_tensor.AdamW` — private API, since removed from torch | pretraining crashes at optimizer creation |
| checkpoint loading | current `torch.load` defaults to `weights_only=True`, which rejects the `argparse.Namespace` stored inside the checkpoints | `main_test`/`main_finetune` cannot open the published weights |
| published checkpoints | store attention as fused `attn.qkv.*`; the current upstream `video_vit.Attention` has separate `attn.q/k/v` | `load_state_dict(strict=False)` silently leaves every attention layer at random init |

**Usability gaps** — things the upstream entry points do not offer and the Sapiens workflow takes for granted:

- `main_pretrain.py` has **no validation loop**; nothing computes a loss on a second dataset during training.
- There is **no hook mechanism**: no way to run a visualization, or anything else, every N iterations without editing the training loop.
- Configuration is ~40 argparse flags; several things that matter (`mean`/`std` normalization, horizontal flip, `pred_t_dim` geometry) are reachable only through the Python constructors, not the CLI.
- Checkpoints are named `checkpoint-%05d.pth`, there is no "last checkpoint" sentinel, and `misc.load_model` silently auto-resumes from whatever it finds in `output_dir`.

**Central goal.** Keep the upstream clone *untouched* and use it as a library, while giving MAE-ST the same mode of use as the Sapiens MAE-ViT encoder — so that switching between the two encoders changes the config, not the workflow:

| Sapiens (mmengine) | here |
|---|---|
| `configs/<name>.py`, tiered, `work_dir` derived from the filename | same |
| VSCode tasks `train` / `compute_loss` / `plot_loss` | same labels, same prompts (config, tmux, resume, dataset, checkpoint) |
| `work_dir/last_checkpoint`, `epoch_N.pth`, `<timestamp>/` per run | same layout |
| `vis_data/scalars.json` + `plot_loss.py` | same format, plus a `val_loss` curve |
| the Sapiens visualization hook (per-sample files, loss in the filename) | `custom/viz_hooks.py::ReconstructionVideoHook` |
| checkpoint every N epochs, visualize every N iterations | `save_every_epochs`, `visualize_every_iterations` (0 disables either) |

Everything that had to be added lives in this directory, and every stack fix is a shim applied from the outside (`custom/compat.py`). `git status` in the clone stays clean.

---

## 2. Quick start

### 2.0 The three directories

The tooling is code only and never holds your data. Three locations, all decided in one place, [`bootstrapper.py`](bootstrapper.py):

| what | default | override |
|---|---|---|
| `TOOLING_ROOT` — this directory: `custom/`, `tools/`, launchers, tasks | wherever it is checked out | — |
| `EXPERIMENTS_ROOT` — **yours**: `configs/`, `work_dirs/`, `checkpoints/` | `~/mae_st_experiments` | `$MAE_ST_EXPERIMENTS` |
| `UPSTREAM_REPO_ROOT` — the pristine `mae_st` clone | `~/mae_st` | `$MAE_ST_UPSTREAM` |
| `DATASETS_ROOT` — datasets | `~/datasets` | `$MAE_ST_DATASETS` |

Every tool accepts bare names and resolves them against these roots: a config name is looked up in `EXPERIMENTS_ROOT/configs/`, a dataset name under `DATASETS_ROOT`, a checkpoint name under `EXPERIMENTS_ROOT/checkpoints/`, a `scalars.json` path under `EXPERIMENTS_ROOT/work_dirs/`. Absolute paths are always accepted as well.

### 2.1 The upstream clone, the experiments directory, the checkpoint

Clone the original repository, unmodified. Its directory **must** be named `mae_st`: the upstream code imports itself as the package `mae_st.*`, and the bootstrapper puts the clone's parent directory on `sys.path` so that works.

```bash
git clone https://github.com/facebookresearch/mae_st.git ~/mae_st
```

Create the experiments directory and put the published pretrain checkpoint (~3.8 GB) under it — the shipped config starts from it:

```bash
mkdir -p ~/mae_st_experiments/{configs,checkpoints}
wget -P ~/mae_st_experiments/checkpoints https://dl.fbaipublicfiles.com/video-mae-200x4-nonorm.pth
```

### 2.2 Environment

Building the Python environment for `mae_st` is its own trial-and-error session (a bundled `environment.yml` that does not work, dependencies frozen at old releases, torch builds that differ by CUDA runtime). Tooling that makes it reproducible — an installer driven by a per-machine manifest recording what to install and why — lives in the [`installer` branch of LCAD-UFES/mae_st_lcad](https://github.com/LCAD-UFES/mae_st_lcad/tree/installer), under `_install/`; follow its README once per machine.

Every command below assumes the resulting environment is active. The VSCode tasks and launchers activate it by name (`.vscode/tasks.json`, `run_training.sh`, `run_tensorboard.sh`); adjust that name if you chose another.

### 2.3 A dataset (the minimum the loader needs)

The upstream `Kinetics` dataset class needs a directory holding `train.csv`, `val.csv` and `test.csv`. Each line is `absolute_path_to_video label` (one space, label an integer, ignored in pretraining but it must parse). Paths cannot contain spaces. Videos should be resized so the short edge is 256 (upstream `DATASET.md`). Which csv is read is decided by the dataset *mode*: `pretrain`→`train.csv`, `val`→`val.csv`, `test`→`test.csv`.

The smallest possible dataset is one video listed in all three files:

```bash
mkdir -p ~/datasets/my_slice && for s in train val test; do
  echo "$HOME/mae_st/demo/demo.avi 0" > ~/datasets/my_slice/$s.csv
done
```

A real 10-video slice of Kinetics-400 is built by `tools/prepare_k400_slice.py` into `~/datasets/kinetics400_slice10/` (10 classes, short edge 256, `labels.txt` with the alphabetical class ids). It downloads only the first ~150 MB of one official tarball, validates each video by decoding it, resizes with PyAV (no `ffmpeg` CLI needed) and writes the csvs — see its `--help` for a larger slice.

### 2.4 An elementary config

A config is a Python file in `EXPERIMENTS_ROOT/configs/`; the drivers execute it and read its top-level names. Because the drivers have already put the tooling on `sys.path`, a config needs no path logic at all: it imports `bootstrapper` for locations and `custom.*` for hooks.

A complete, fully commented one ships here as [`EXAMPLE_CONFIG_overfit_k400_slice10-vitl_nonorm.py`](EXAMPLE_CONFIG_overfit_k400_slice10-vitl_nonorm.py) — copy it into `EXPERIMENTS_ROOT/configs/`, drop the `EXAMPLE_CONFIG_` prefix and edit from there (the `work_dir` follows the file name, so renaming it names the experiment). It is the only config in this repository; the ones you actually run live in your experiments directory. This is the same thing cut down to the minimum:

```python
# ~/mae_st_experiments/configs/my_first_test.py
import os.path as osp
import bootstrapper
from custom.viz_hooks import ReconstructionVideoHook

lr = min_lr = 1e-5; warmup_epochs = 0        # constant LR (see §3.8)
mask_ratio = 0.9
max_epochs = 20
save_every_epochs = 5                        # 0 = only the final checkpoint
visualize_every_iterations = 25              # 0 = never during training
batch_size = 2; repeat_aug = 2; seed = 0

_common = dict(path_to_data_dir="kinetics400_slice10",   # a name under DATASETS_ROOT
               sampling_rate=4, num_frames=16)
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

Open this directory in VSCode (alone, or as one folder of a multi-root workspace together with the experiments directory and the clone). The tasks live in `.vscode/tasks.json` here and prompt for bare names (`Terminal → Run Task…`, or the TASKS panel):

| task | asks for | does |
|---|---|---|
| `train` | config name, tmux?, resume-or-restart | trains; writes `EXPERIMENTS_ROOT/work_dirs/mae_st_overfit_tests/<config>/` |
| `compute_loss` | config name, checkpoint (blank = last), dataset name | loss per video + visualizations for one checkpoint |
| `plot_loss` | `scalars.json` path relative to `work_dirs/` | PNG with train and val loss curves |
| `tensorboard` | — | dashboard over *all* work_dirs, in the foreground (stop with Ctrl+C) |

Or from the shell, from any directory:

```bash
python tools/train.py my_first_test.py                       # add --resume to continue
python tools/compute_loss.py my_first_test.py --dataset kinetics400_slice10
python tools/plot_loss.py mae_st_overfit_tests/my_first_test/<ts>/vis_data/scalars.json
```

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
│       └── epoch0005_iter000025/<video>_loss0.0823.mp4   [original | masked | reconstruction | error]
└── compute_loss/<timestamp>-epoch_20/
    ├── <timestamp>.log, <timestamp>.json   mean loss + per-sample losses
    └── vis_data/eval/<video>_loss0.0801.mp4
```

Checkpoints are ~3.8 GB each (ViT-L plus optimizer state) — mind `save_every_epochs` and `max_keep_ckpts` on long runs.

---

## 3. How it is built

### 3.1 Code map

```
   ~/mae_st  (upstream clone, never edited, imported as package mae_st.*)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ models_mae.py   util/kinetics.py   util/misc.py   util/lr_sched.py   │
   │ engine_pretrain.py (copied, see custom/engine.py)   util/decoder/    │
   └──────▲───────────────▲───────────────▲───────────────▲───────────────┘
          │ build model    │ build dataset │ AdamW groups, │ lr per iter
          │                │               │ NativeScaler  │
 <mae_st_tooling>  (code)  │               │               │
 ┌────────┼────────────────┼───────────────┼───────────────┼─────────────┐
 │ bootstrapper.py ── the setup: locations (TOOLING/EXPERIMENTS/UPSTREAM/ │
 │                    DATASETS roots), sys.path, compat shims; imported   │
 │                    first by every driver and by every config           │
 │ custom/compat.py ── shims: torch.fb stub, sklearn stub, torch.load     │
 │                     allowlist, PyAV replacement for the removed        │
 │                     torchvision video API                              │
 │ custom/experiment.py ── config loading, run dirs, console tee, fixed   │
 │                     val clips, deterministic val loss, hooks runner,   │
 │                     checkpoint save/resume/sentinel, scalars.json      │
 │ custom/engine.py ── verbatim copy of train_one_epoch + after_iter()    │
 │ custom/checkpoints.py ── loads published checkpoints (fused qkv split) │
 │ custom/viz_hooks.py ── ReconstructionVideoHook (per-sample mp4)        │
 │ tools/train.py, compute_loss.py, plot_loss.py                          │
 │ run_training.sh (tmux, resume), run_tensorboard.sh, .vscode/tasks.json │
 │ tools/prepare_k400_slice.py, tools/run_test_smoke.py                   │
 │ EXAMPLE_CONFIG_*.py ── template to copy into EXPERIMENTS_ROOT/configs/ │
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

The whole initial setup, as a side effect of `import bootstrapper`: it computes the four roots (this directory; `EXPERIMENTS_ROOT`, `UPSTREAM_REPO_ROOT`, `DATASETS_ROOT` from their defaults or the `MAE_ST_*` environment variables), puts the tooling and the upstream clone's *parent* directory on `sys.path` (the clone's directory name must stay `mae_st`), and applies the compatibility shims. It also provides `work_dir_for(config_name)` (`EXPERIMENTS_ROOT/work_dirs/mae_st_overfit_tests/<name>`) and `resolve(path, base)`, the bare-name resolution every tool and prompt relies on. `python bootstrapper.py` prints the roots in effect.

### 3.3 `custom/compat.py`

Four shims, applied from outside the clone:

1. **`torch.fb.rendezvous.zeus`** — an empty module registered in `sys.modules` before `util/misc.py` imports it. (The stub carries a `__spec__` because `torch._dynamo` walks `sys.modules` with `find_spec`, which raises on modules without one.)
2. **`sklearn`** — same trick; the stubbed `average_precision_score` raises if ever called, so a multi-label evaluation fails loudly instead of silently.
3. **`torch.load` allowlist** — `torch.serialization.add_safe_globals([argparse.Namespace])` so checkpoints that embed their training args load under `weights_only=True`.
4. **Video decoding** — `torchvision.io._probe_video_from_memory` and `_read_video_from_memory` reimplemented on PyAV and monkeypatched onto the `torchvision.io` module object, which upstream's `decoder.py` holds a reference to. Honors the `pts` range upstream uses for selective decoding. DataLoader workers are forked (Linux), so they inherit the patch.

`tools/run_test_smoke.py` contains the same four shims written inline with a long commentary — read it first if you want to understand *why* each exists.

### 3.4 `custom/experiment.py`

Everything `train.py` and `compute_loss.py` share.

`load_config(name_or_path)` resolves the config (bare names in `EXPERIMENTS_ROOT/configs/`), runs it with `runpy` and returns its namespace as a dict (required keys are checked). `make_run_dir()` creates `work_dir/<ts>/vis_data/` and copies the config verbatim to `work_dir/` and `work_dir/<ts>/`. `install_tee()` mirrors stdout/stderr into `<ts>.log` — upstream prints go through the builtin `print`, so its progress lines are captured too. `build_dataset()` resolves `path_to_data_dir` against `DATASETS_ROOT`.

`load_fixed_clips(dataset, n, seed)` decodes the validation clips **once**, under a fixed seed, and keeps them in memory. Three RNGs feed a Kinetics sample (Python `random` for the temporal start, `numpy` for crops/flips, torch for the MAE mask); re-seeding them every epoch would also reset the training sampler's shuffle. Freezing the clips and forking only the torch RNG around each evaluation (`fork_seeded`) makes the validation masks identical from epoch to epoch without touching training randomness. That is what makes `val_loss` comparable over time, and why `compute_loss` on `epoch_N.pth` reproduces exactly the `val_loss` train.py logged. `masked_loss()` is that evaluation; `run_hooks()` runs the hooks under the same envelope and merges their `{stem: loss}` results.

Checkpoints: `save_checkpoint()` writes `epoch_N.pth` with the upstream dict keys (`model`, `optimizer`, `epoch`, `scaler`, `args`) plus `config` and `run_dir`, rewrites `last_checkpoint`, prunes to `max_keep_ckpts`. `resolve_checkpoint()` has the Sapiens semantics: explicit path (or a name under `checkpoints/`) wins, else the sentinel, else a loud error. `load_resume_checkpoint()` restores model/optimizer/scaler and returns the next epoch to run.

### 3.5 `custom/engine.py`

A verbatim copy of upstream `engine_pretrain.train_one_epoch` — same meters, same per-iteration `lr_sched.adjust_learning_rate`, same NaN guard — with one addition: an `after_iter(global_iter)` callback at the end of every optimizer step. It is the hook point upstream lacks and the only way to visualize every N *iterations* without editing the clone. When upstream changes, re-diff this file against `engine_pretrain.py`.

One subtlety it inherits: on a non-finite loss the upstream code deletes `num_checkpoint_del` files chosen by `misc.get_last_checkpoint`, whose filter is `"checkpoint" in filename` — which matches our `last_checkpoint` sentinel. `train.py` therefore passes `num_checkpoint_del=0`; the engine then only raises.

### 3.6 `custom/checkpoints.py`

`load_mae_checkpoint(model, path)`: loads a pretrain checkpoint into a `models_mae` model, converting the fused `attn.qkv.{weight,bias}` of the published checkpoints into the separate `attn.q/k/v` that upstream `video_vit.Attention` now uses (a 3-way chunk along dim 0, the timm layout), for encoder and decoder blocks alike. It raises if any model key is left uninitialized — the failure mode it prevents is a model that loads "successfully" with random attention weights. Our own `epoch_N.pth` files already have split keys and pass through unchanged.

### 3.7 `custom/viz_hooks.py`

The hook contract: a callable `hook(model, samples, stems, tag, run_dir) -> {stem: loss}` where `samples` is a normalized `[B, 3, T, H, W]` batch on the model's device, `stems` names each sample, `tag` labels the moment (`epoch0005_iter000025`, `eval`) and outputs go under `run_dir/vis_data/<tag>/`. Hooks are instantiated **in the config**, so a new visualization is a new callable listed there — no driver changes.

`ReconstructionVideoHook` writes one mp4 per sample, four panels side by side: original, masked input (masked patches in gray), reconstruction, and the per-pixel error. The reconstruction panel is the *composite* the MAE papers show — true pixels at visible patches, prediction only at masked ones — because the loss only ever penalizes masked patches and the decoder's output at visible positions is untrained noise. The filename carries the per-sample masked-region loss (`<stem>_loss0.0823.mp4`), computed with the same formula as `forward_loss` reduced per sample. `norm_pix_loss` checkpoints are handled (prediction is de-normalized with the target's per-patch statistics before rendering).

### 3.8 `tools/train.py`

```
python tools/train.py <config> [--resume] [--tensorboard on|off]
```

Owns the epoch loop; upstream supplies the parts. Per config it builds the `Kinetics` train dataset (mode `pretrain`) and a plain `DataLoader` (`drop_last=False` — upstream's `True` would drop videos from a 10-video set), the fixed validation clips, the model (initialized from `load_from` when not resuming), `misc.add_weight_decay` parameter groups, `torch.optim.AdamW(betas=(0.9, 0.95))` (the public replacement for the private optimizer upstream uses) and `misc.NativeScalerWithGradNormCount`. It assembles the `argparse.Namespace` the upstream engine and `lr_sched` read (`accum_iter`, `mask_ratio`, `clip_grad`, `repeat_aug`, `lr`, `min_lr`, `warmup_epochs`, `epochs`, `output_dir`, `num_checkpoint_del=0`).

Each epoch: `custom.engine.train_one_epoch` (with the hooks firing from `after_iter` when `visualize_every_iterations > 0`), then `val_loss` on the fixed clips, a line appended to `scalars.json`, and a checkpoint when `save_every_epochs` divides the epoch count (always on the last epoch). Epoch numbers shown to the user are *epochs completed*, 1-based; the engine internally receives the upstream 0-based index.

**Learning rate.** Upstream's `lr_sched` applies linear warmup then half-cycle cosine from `lr` to `min_lr`, unconditionally. For a constant rate set `lr == min_lr` and `warmup_epochs = 0`. `lr` may be left `None` to use `blr` with upstream's linear scaling rule (`blr × effective batch / 256`).

**Resume.** `--resume` reads `work_dir/last_checkpoint`, restores everything, and starts a *new* `<ts>/` run directory (own log, config copy, tb), pre-filled with the previous run's `scalars.json` records so the curve stays continuous. `run_training.sh` adds the flag only when the user chose "Continuar" *and* the sentinel exists.

**TensorBoard.** Events go to `<run>/tb` by default (`flush_secs=10`), which is what allows the dashboard to be attached later; `--tensorboard off` disables the writer (`scalars.json` is always written).

### 3.9 `tools/compute_loss.py`

```
python tools/compute_loss.py <config> --dataset DIR [--checkpoint PATH] [--mode val|pretrain|test]
```

Inference over a dataset directory with one checkpoint (blank → `last_checkpoint`; an explicit path or a name under `checkpoints/` may be a published fused-qkv checkpoint). Uses the config's `val_dataset` settings with `path_to_data_dir` replaced by `--dataset` and the csv chosen by `--mode`. Runs the config's hooks (or a default `ReconstructionVideoHook`) over every clip with the same fixed masking as training's `val_loss`, prints the per-sample table and the mean, and writes `work_dir/compute_loss/<ts>-<checkpoint_stem>/` with `<ts>.log`, `<ts>.json` and `vis_data/eval/`.

Because the masks and clips are fixed, the number it reports is a property of the checkpoint and the dataset only, not of the run — so it can be compared across checkpoints, and against the `val_loss` column of `scalars.json`. Two uses follow from that: pointing `--dataset` at the training set gives the training loss measured *without* the training-time augmentation randomness (the per-iteration `loss` in `scalars.json` is averaged over random clips and masks, so it is noisier and not directly comparable); pointing it at a dataset the model never saw gives the loss on unseen data, and the difference between the two is the generalization gap of that checkpoint.

### 3.10 `tools/plot_loss.py`

```
python tools/plot_loss.py <scalars.json> [--output PATH]
```

Reads the JSON-lines file (a path, or one relative to `EXPERIMENTS_ROOT/work_dirs/`; a run directory or a config's work_dir is accepted too, the latter plotting its newest run — which is what the task's default prompt value does), plots `loss` and (when present) `val_loss` against epoch, saves `<input>_loss_curve.png` next to the input. Malformed lines are skipped with a warning.

### 3.11 `run_training.sh`, `run_tensorboard.sh`, `.vscode/`

`run_training.sh <config> <Sim|Não> <resume_choice>` is the launcher behind the `train` task, same design as the Sapiens one: every decision arrives as an argument (answered by VSCode pick lists before the script runs, so nothing ever blocks on a `read` inside a possibly flaky SSH session); it asks the tooling to resolve the config and its `work_dir`, decides `--resume` from the `last_checkpoint` sentinel, writes the inner command to a temp script and runs it either inside a detached tmux session named `<config>-<timestamp>` (survives connection drops; `tmux attach -t …`) or in the foreground. It holds no paths of its own — everything comes from the bootstrapper.

`run_tensorboard.sh [logdir] [port]` runs TensorBoard in the *foreground* over all of `EXPERIMENTS_ROOT/work_dirs/` (so runs from every config, finished or in progress, can be overlaid), bound to localhost only — TensorBoard has no authentication. Started from the `tensorboard` task it is stoppable from the task panel, and VSCode's Remote-SSH detects the port and offers to open it (`.vscode/settings.json` labels port 6006). Independent of the editor, a tunnel from your machine reaches the same server: `autossh -M 0 -N -L 6006:localhost:6006 <user>@<host>`.

`.vscode/tasks.json` defines the four tasks; their commands use `${workspaceFolder}` (this directory) and their prompts take bare names, so the file contains no location of the experiments directory. Labels intentionally carry no `.py`.

### 3.12 `tools/prepare_k400_slice.py` and `tools/run_test_smoke.py`

`prepare_k400_slice.py` builds the Kinetics slice: annotations and a partial download of one validation tarball from the CVD Foundation S3 mirror (tar+gzip is a stream, so the first 150 MB already hold dozens of complete videos), a full-decode check to drop the truncated tail, selection of N videos of distinct classes, PyAV resize to short edge 256 with libx264, alphabetical class ids in `labels.txt`, and the same videos written to all three csvs (`--out-dir`, `--raw-dir`, `--n-videos`, `--partial-mb`).

`run_test_smoke.py` is the first thing that ran against the pristine clone: it drives upstream's own `run_test.py` end to end with the four shims written inline and commented at length. It is kept as the didactic record of what breaks and why; the reusable form is `custom/compat.py`. It reads the same `MAE_ST_*` environment variables as the bootstrapper but does not import it, so that the shims it demonstrates are really the ones doing the work.
