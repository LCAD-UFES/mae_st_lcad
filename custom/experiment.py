"""Shared plumbing for the tools/ drivers (train.py, compute_loss.py).

Importing this module imports `bootstrapper`, which does the environment
setup (sys.path for the tooling and the upstream clone, compatibility shims)
BEFORE any mae_st import -- which is why the drivers import this first and
only then touch mae_st.

Conventions implemented here (mirroring the sapiens/mmengine workspace),
all under EXPERIMENTS_ROOT:
  work_dir/<config>.py           verbatim config copy
  work_dir/last_checkpoint        one line, absolute path of the newest .pth
  work_dir/epoch_N.pth            N = epochs completed (1-based); pruned to max_keep
  work_dir/<ts>/                  one dir per driver invocation
  work_dir/<ts>/<ts>.log          console mirror (stdout + stderr)
  work_dir/<ts>/vis_data/scalars.json   JSON-lines {epoch, loss, val_loss, lr, time}
"""

import glob
import json
import os
import os.path as osp
import random
import re
import runpy
import shutil
import sys
import time

import bootstrapper as bs  # noqa: F401  (side effects: sys.path + compat shims)
import numpy as np
import torch

# ---------------------------------------------------------------------------
# config / run directory / logging
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = ("work_dir", "model_name", "model_kwargs", "val_dataset",
                        "hooks", "mask_ratio", "seed")


def load_config(path):
    """Execute a config file and return its namespace as a dict. A bare name
    is looked up in EXPERIMENTS_ROOT/configs/."""
    path = bs.resolve(path, bs.CONFIGS_DIR)
    ns = runpy.run_path(path)
    cfg = {k: v for k, v in ns.items() if not k.startswith("__")}
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"config {path} lacks required keys: {missing}")
    cfg["__config_path__"] = path
    cfg["work_dir"] = osp.expanduser(cfg["work_dir"])
    return cfg


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(work_dir, config_path, ts):
    """work_dir/<ts>/vis_data, plus verbatim config copies at work_dir root
    and inside the run dir (sapiens layout)."""
    run_dir = osp.join(work_dir, ts)
    os.makedirs(osp.join(run_dir, "vis_data"), exist_ok=True)
    name = osp.basename(config_path)
    shutil.copy2(config_path, osp.join(work_dir, name))
    shutil.copy2(config_path, osp.join(run_dir, name))
    return run_dir


class Tee:
    """Mirror everything written to a stream into a log file, live."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ...
        return getattr(self.stream, name)


def install_tee(log_path):
    """Route stdout and stderr through a Tee into log_path. Returns a
    closer to call in `finally`. Upstream prints go through builtins.print
    (we never call misc.init_distributed_mode), so they are captured too."""
    log_file = open(log_path, "a", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(orig_out, log_file), Tee(orig_err, log_file)

    def close():
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()

    return close


def append_scalar(run_dir, record):
    with open(osp.join(run_dir, "vis_data", "scalars.json"), "a") as f:
        f.write(json.dumps(record) + "\n")


def read_scalars(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# model / data
# ---------------------------------------------------------------------------


def build_model(cfg, device):
    from mae_st import models_mae

    model = models_mae.__dict__[cfg["model_name"]](**cfg["model_kwargs"])
    return model.to(device)


def build_dataset(dataset_kwargs):
    from mae_st.util.kinetics import Kinetics

    kwargs = dict(dataset_kwargs)
    kwargs["path_to_data_dir"] = bs.resolve(kwargs["path_to_data_dir"], bs.DATASETS_ROOT)
    return Kinetics(**kwargs)


def video_stem(dataset, index):
    return osp.splitext(osp.basename(dataset._path_to_videos[index]))[0]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fixed_clips(dataset, max_videos, seed):
    """Decode clips ONCE under a fixed seed and keep them as a CPU tensor.

    Three RNGs feed a Kinetics sample (python `random` for the temporal
    start, `np.random` for crops/flips, torch for the MAE mask). Re-seeding
    them every evaluation would also reset the training sampler's shuffle,
    so instead the clips are frozen here and only the torch RNG is forked
    around each evaluation (see fork_seeded()). Returns
    (clips [N, 3, T, H, W], stems)."""
    n = len(dataset) if max_videos is None else min(len(dataset), max_videos)
    seed_everything(seed)
    clips, stems = [], []
    for i in range(n):
        frames = dataset[i][0]  # [repeat_aug, 3, T, H, W]
        stem = video_stem(dataset, i)
        for rep in range(frames.shape[0]):
            clips.append(frames[rep])
            stems.append(stem if frames.shape[0] == 1 else f"{stem}_rep{rep}")
    return torch.stack(clips), stems


class fork_seeded:
    """Context: run with torch RNG seeded to `seed`, then restore the RNG
    state so training randomness continues untouched."""

    def __init__(self, seed, device):
        self.seed = seed
        if device.type == "cuda":
            # torch.device("cuda") has index None; fork_rng needs a real index
            idx = device.index if device.index is not None else torch.cuda.current_device()
            self.devices = [idx]
        else:
            self.devices = []

    def __enter__(self):
        self._ctx = torch.random.fork_rng(devices=self.devices)
        self._ctx.__enter__()
        torch.manual_seed(self.seed)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


@torch.no_grad()
def masked_loss(model, clips, mask_ratio, batch_size, device, seed):
    """Mean MAE loss over `clips` with masks fixed by `seed` (identical on
    every call -> comparable across epochs). Leaves the model in eval mode."""
    model.eval()
    total, count = 0.0, 0
    with fork_seeded(seed, device):
        for start in range(0, len(clips), batch_size):
            batch = clips[start:start + batch_size].to(device, non_blocking=True)
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
            total += float(loss) * len(batch)
            count += len(batch)
    return total / count


@torch.no_grad()
def run_hooks(hooks, model, clips, stems, tag, run_dir, batch_size, device, seed):
    """Invoke every hook over `clips` in batches, under the same fixed
    masking as masked_loss(). Returns merged {stem: loss}. Leaves the model
    in eval mode -- callers that train must restore train mode."""
    model.eval()
    per_sample = {}
    with fork_seeded(seed, device):
        for start in range(0, len(clips), batch_size):
            batch = clips[start:start + batch_size].to(device, non_blocking=True)
            batch_stems = stems[start:start + batch_size]
            for hook in hooks:
                result = hook(model, batch, batch_stems, tag, run_dir)
                if isinstance(result, dict):
                    per_sample.update(result)
    return per_sample


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

SENTINEL = "last_checkpoint"


def read_last_checkpoint(work_dir):
    sentinel = osp.join(work_dir, SENTINEL)
    if not osp.isfile(sentinel):
        return None
    with open(sentinel) as f:
        path = f.read().strip()
    if not osp.isfile(path):
        raise FileNotFoundError(f"{sentinel} points at a missing file: {path}")
    return path


def resolve_checkpoint(work_dir, explicit):
    """sapiens semantics: a non-empty explicit path wins (a bare name is
    looked up in EXPERIMENTS_ROOT/checkpoints/); else the sentinel; else a
    loud error (never silently fall back to random weights)."""
    if explicit is not None and explicit.strip():
        return bs.resolve(explicit, bs.CHECKPOINTS_DIR)
    path = read_last_checkpoint(work_dir)
    if path is None:
        raise FileNotFoundError(
            f"no --checkpoint given and no {SENTINEL} in {work_dir}; "
            "train first or pass --checkpoint explicitly")
    return path


def _epoch_of(path):
    m = re.search(r"epoch_(\d+)\.pth$", path)
    return int(m.group(1)) if m else -1


def save_checkpoint(work_dir, epochs_done, model, optimizer, loss_scaler, args,
                    config_path, run_dir, max_keep):
    """Write work_dir/epoch_<epochs_done>.pth (upstream dict keys plus our
    provenance), refresh the sentinel, prune to `max_keep` newest."""
    path = osp.join(work_dir, f"epoch_{epochs_done}.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epochs_done - 1,  # upstream stores the 0-based index of the completed epoch
        "scaler": loss_scaler.state_dict(),
        "args": args,
        "config": config_path,
        "run_dir": run_dir,
    }, path)
    with open(osp.join(work_dir, SENTINEL), "w") as f:
        f.write(osp.abspath(path) + "\n")
    kept = sorted(glob.glob(osp.join(work_dir, "epoch_*.pth")), key=_epoch_of)
    for old in kept[:-max_keep] if max_keep > 0 else []:
        os.remove(old)
        print(f"[checkpoint] pruned {old}")
    print(f"[checkpoint] saved {path}")
    return path


def load_resume_checkpoint(path, model, optimizer, loss_scaler):
    """Restore a checkpoint written by save_checkpoint(); returns
    (start_epoch (0-based, the next epoch to run), checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    loss_scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[checkpoint] resumed from {path}: {start_epoch} epochs completed")
    return start_epoch, ckpt
