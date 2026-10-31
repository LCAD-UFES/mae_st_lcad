# Contributing to mae_st_tooling

How the tooling is built. [README.md](README.md) is how to use it; repository-wide conventions (where a change belongs, tests, commits, branches) are in `main`'s [CONTRIBUTING](https://github.com/LCAD-UFES/mae_st_lcad/blob/main/CONTRIBUTING.md).

## 1. Why a tooling at all

`main_pretrain.py` and its siblings are single commands with ~40 argparse flags: no versionable config, no record of what ran, no validation loss, no place to plug a visualization into the loop. Several things that matter here (crop size and position, resampling, normalization, flip, `pred_t_dim`) are reachable only through the Python constructors. Upstream checkpoints are named `checkpoint-%05d.pth`, there is no "last checkpoint" sentinel, and `misc.load_model` silently auto-resumes from whatever it finds in `output_dir`.

The tooling uses `main` as a library and gives it the workflow this group already uses for the Sapiens MAE-ViT encoder (mmengine), so that moving between the two changes the config, not the workflow:

| Sapiens | here |
|---|---|
| `configs/<name>.py`, `work_dir` derived from the file name | same |
| VSCode tasks `train` / `compute_loss` / `plot_loss` | same labels and prompts |
| `work_dir/last_checkpoint`, `epoch_N.pth`, `<timestamp>/` per run | same layout |
| `vis_data/scalars.json` + `plot_loss.py` | same format, plus `val_loss` |
| visualization hook writing per-sample files, loss in the file name | `custom/viz_hooks.py::ReconstructionVideoHook` |

## 2. Code map

```
<MAE_ST_REPO_ROOT>  mae_st_lcad main, imported as the package mae_st
  models_mae.py  util/kinetics.py  util/misc.py  util/lr_sched.py  engine_pretrain.py
        ▲ model, dataset, AdamW groups, NativeScaler, lr per iteration, train step
<mae_st_tooling>
  bootstrapper.py          locations, sys.path, the mae_st package; imported first everywhere
  custom/experiment.py     config loading, run dirs, console tee, fixed val clips,
                           deterministic val loss, hooks runner, checkpoints, scalars.json
  custom/checkpoints.py    loads published checkpoints (fused qkv split)
  custom/viz_hooks.py      ReconstructionVideoHook
  tools/                   train.py, compute_loss.py, plot_loss.py, prepare_dataset.py,
                           prepare_k400_slice.py
  env.sh, run_training.sh, run_tool.sh, run_tensorboard.sh, .vscode/tasks.json
  _install/                the environment installer (its own README and DESIGN)
  tests/                   unittest, CPU
        │ reads configs/, writes work_dirs/, loads checkpoints/
<EXPERIMENTS_ROOT>  yours
```

The one rule: `bootstrapper` is imported before anything imports `mae_st`. `custom/experiment.py` imports it at the top, which is why every driver starts with `from custom import experiment`.

## 3. Components

### `bootstrapper.py`

The whole setup, as a side effect of `import bootstrapper`: the four roots (this directory; `EXPERIMENTS_ROOT`, `MAE_ST_REPO_ROOT`, `DATASETS_ROOT` from defaults or `MAE_ST_*` variables), the tooling on `sys.path`, and the mae_st clone registered as the package `mae_st` whatever its directory name; a missing clone fails at import naming the variable to set. It also provides `work_dir_for(config_name)` and `resolve(path, base)`, the bare-name resolution every tool and prompt relies on.

### `custom/experiment.py`

What `train.py` and `compute_loss.py` share. `load_config` runs the config with `runpy` and checks its required keys. `make_run_dir` creates `work_dir/<ts>/vis_data/` and copies the config verbatim. `install_tee` mirrors stdout/stderr into `<ts>.log`, upstream's prints included. `build_dataset` resolves `path_to_data_dir` against `DATASETS_ROOT`.

**Deterministic validation.** `load_fixed_clips` decodes the validation clips once, under a fixed seed, and keeps them in memory. Three RNGs feed a sample (Python `random` for the temporal start, numpy for crops and flips, torch for the mask); re-seeding them every epoch would also reset the training shuffle. Freezing the clips and forking only the torch RNG around each evaluation (`fork_seeded`) gives identical validation masks from epoch to epoch without touching training randomness. That makes `val_loss` comparable over time, and is why `compute_loss` on `epoch_N.pth` reproduces exactly the `val_loss` that `train.py` logged.

**Checkpoints.** `save_checkpoint` writes `epoch_N.pth` with upstream's keys (`model`, `optimizer`, `epoch`, `scaler`, `args`) plus `config` and `run_dir`, rewrites `last_checkpoint` and prunes to `max_keep_ckpts`. `resolve_checkpoint` is the one place that answers "which weights": `last` is the sentinel (error if absent), a path or name is itself, nothing is the config's `load_from`; it also returns which case it was. `load_resume_checkpoint` refuses checkpoints this tooling did not write.

### `tools/train.py`

Owns the epoch loop; `main` supplies the parts: the `Kinetics` dataset and a `DataLoader` (`drop_last=False`, since upstream's `True` would drop clips from a small set), the model (from `load_from` unless resuming), `misc.add_weight_decay` groups, `torch.optim.AdamW`, `NativeScalerWithGradNormCount`, and the `argparse.Namespace` the engine and `lr_sched` read. Each epoch runs `engine_pretrain.train_one_epoch` with hooks firing from `after_iter`, then `val_loss`, a line in `scalars.json`, and a checkpoint every `save_every_epochs` and on the last epoch.

- **Learning rate.** `lr_sched` applies linear warmup then half-cycle cosine from `lr` to `min_lr`, unconditionally; `lr == min_lr` with `warmup_epochs = 0` is a constant rate. `lr = None` uses `blr` with upstream's linear scaling.
- **Gradient accumulation.** The engine steps every `accum_iter` iterations, counting from 0 each epoch. An `accum_iter` larger than the iterations per epoch (clips / `batch_size`) never steps, so it is refused; one that does not divide them is a warning. In the overfit configs `accum_iter` equals the iterations per epoch: one step per epoch on the full-set gradient.
- **Resume** restores model, optimizer, scaler and epoch, and opens a new `<ts>/` pre-filled with the previous `scalars.json`, so the curve stays continuous. A checkpoint without optimizer state or with fused `attn.qkv` (a published one) is refused with a pointer to `load_from`.
- **`num_checkpoint_del=0`** is load-bearing: on a non-finite loss the engine deletes files chosen by `misc.get_last_checkpoint`, whose filter `"checkpoint" in filename` matches the `last_checkpoint` sentinel.
- **Weights & Biases** is optional: a config with `wandb = dict(project=..., mode="offline")` logs per-epoch `loss`, `val_loss`, `lr`. **TensorBoard** events go to `<run>/tb` unless `--tensorboard off`.

### `tools/compute_loss.py`

One checkpoint over one dataset, with the same checkpoint scheme as `train.py` (omitting it evaluates `load_from`, a one-command "before training" baseline). It uses the config's `val_dataset` with `path_to_data_dir` replaced by `--dataset`; the csv is `--csv`, else the config dataset's `csv_file` for that `--mode`, else the mode's default. The masks and clips are fixed as in training, so the number is a property of checkpoint and dataset only: pointed at the training set it measures the training loss without augmentation noise, at unseen data the generalization gap.

### `tools/plot_loss.py`

Plots `loss` and `val_loss` from a `scalars.json`, a run dir or a config's work_dir (its newest run), into a PNG next to the input.

### `tools/prepare_dataset.py`, `tools/prepare_k400_slice.py`

`prepare_dataset.py` writes the csvs for a folder of mp4s. Without options the folder is one slice, listed in `train.csv`, `val.csv` and `test.csv`; `--split prefix` splits it by the `train_`/`val_`/`test_` prefixes; `--prefix P ...` writes one `P.csv` per lsv2 slice. It mirrors lidar_sweep_viewer2 without depending on it: an lsv2 job is a slice, its `prefix` names the files, and here it names the csv. `prepare_k400_slice.py` builds a 10-video Kinetics-400 slice from a partial download of one official tarball.

### `custom/viz_hooks.py`

A hook is a callable `hook(model, samples, stems, tag, run_dir) -> {stem: loss}`, instantiated in the config, so a new visualization needs no driver change. `ReconstructionVideoHook` writes one mp4 per sample with four panels at input resolution: original, masked input, reconstruction, per-pixel error. Panels are stacked vertically for frames wider than tall (`layout="auto"`), so a 64 x 2650 clip becomes a 256 x 2650 video. The reconstruction is the composite the MAE papers show (true pixels at visible patches, prediction at masked ones), because the loss only covers masked patches. The file name carries the per-sample masked loss. Its `mean`/`std` must be the dataset's.

### `custom/checkpoints.py`

`load_mae_checkpoint` loads a pretrain checkpoint, splitting the fused `attn.qkv` of the published checkpoints into the separate `attn.q/k/v` that `video_vit.Attention` uses, and raises if any model key stays uninitialized: the failure it prevents is a model that loads "successfully" with random attention.

### Launchers and VSCode

`env.sh` finds conda (`CONDA_EXE`, else `conda info --base`) and activates `$MAE_ST_CONDA_ENV` (default `mae_st`). `run_tool.sh <tool> [args]` backs `compute_loss` and `plot_loss`. `run_training.sh <config> <Sim|Não> [checkpoint]` gets every decision as an argument, answered by the VSCode prompts before it runs, so nothing blocks on input inside a fragile SSH session; it runs `train.py` in a detached tmux session or in the foreground and decides nothing about checkpoints. `run_tensorboard.sh [logdir] [port]` serves all work_dirs on localhost only (TensorBoard has no authentication). `.vscode/tasks.json` spells the tooling location as `~/mae_st_tooling`, because `${workspaceFolder}` does not resolve in a multi-root workspace.

## 4. Tests

`python -m unittest discover -s tests -t .`, with `$MAE_ST_REPO` pointing at the `main` clone if it is not at the default. A new tool option comes with a test; `tests/test_train_e2e.py` runs the full train → compute_loss path on two synthetic videos.
