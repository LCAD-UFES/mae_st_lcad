"""Plot training (and validation) loss from a scalars.json written by train.py.

    python tools/plot_loss.py <scalars.json> [--output PATH]

The input is a path, or a path relative to EXPERIMENTS_ROOT/work_dirs/ (e.g.
`mae_st_overfit_tests/<config>/<ts>/vis_data/scalars.json`). A directory is
accepted too: a run dir (`.../<ts>/`) plots its own scalars.json, a config's
work_dir (`.../<config>/`) plots its most recent run. The file is JSON-lines:
one record per epoch with at least {epoch, loss}; records carrying val_loss
add a second curve. Malformed lines are skipped with a warning. Output
defaults to <input without extension>_loss_curve.png next to the input.
"""

import argparse
import glob
import json
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import bootstrapper as bs  # noqa: E402


def find_scalars(path):
    """Accept the file itself, a run dir, or a config work_dir (newest run)."""
    if osp.isfile(path):
        return path
    candidates = [osp.join(path, "vis_data", "scalars.json")]
    candidates += sorted(glob.glob(osp.join(path, "*", "vis_data", "scalars.json")),
                         reverse=True)  # run dirs are timestamps: newest first
    for c in candidates:
        if osp.isfile(c):
            return c
    raise FileNotFoundError(f"no scalars.json found at or under {path}")


def load_records(path):
    records = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {n} (invalid JSON: {e})")
                continue
            if "epoch" not in rec or "loss" not in rec:
                print(f"Warning: skipping line {n} (no epoch/loss)")
                continue
            records.append(rec)
    if not records:
        raise ValueError(f"no valid (epoch, loss) records in {path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_file = find_scalars(bs.resolve(args.log_file, bs.WORK_DIRS))
    print(f"Reading: {log_file}")
    records = load_records(log_file)
    epochs = [r["epoch"] for r in records]
    losses = [r["loss"] for r in records]
    val = [(r["epoch"], r["val_loss"]) for r in records if r.get("val_loss") is not None]
    print(f"Loaded {len(records)} entries (epoch range: {min(epochs)} to {max(epochs)}).")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", markersize=3, linewidth=1, label="train loss")
    if val:
        ax.plot([e for e, _ in val], [v for _, v in val], marker="s", markersize=3,
                linewidth=1, label="val loss (fixed clips/masks)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = "Training / validation loss" if val else "Training loss"
    ax.set_title(f"{title}\n({osp.basename(log_file)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = args.output or osp.splitext(log_file)[0] + "_loss_curve.png"
    fig.savefig(output, dpi=150)
    print(f"Plot saved to: {output}")


if __name__ == "__main__":
    main()
