"""Compute the MAE-ST loss of a checkpoint over a dataset, with visualizations.

    python tools/compute_loss.py <config> --dataset DIR [--checkpoint PATH]
                                 [--mode val] [--batch-size N]

  config       the training config (model geometry, hooks, work_dir, seed);
               a path, or a bare name looked up in EXPERIMENTS_ROOT/configs/
  --dataset    a Kinetics data dir holding {train,val,test}.csv (a bare name
               is looked up under DATASETS_ROOT); which csv is read depends
               on --mode (default "val" -> val.csv). Point it at the training
               set to recompute train loss through the identical code path,
               or at a held-out set for the generalization gap.
  --checkpoint which weights to evaluate, same scheme as train.py:
               `last` = the newest checkpoint of this config; a path (or a
               bare name under EXPERIMENTS_ROOT/checkpoints/), which may be a
               published, fused-qkv pretrain checkpoint; omitted = the
               config's `load_from`, i.e. the point training would start from.

Writes work_dir/compute_loss/<ts>-<checkpoint_stem>/:
  <ts>.log        console mirror
  <ts>.json       {config, checkpoint, dataset, mode, mask_ratio, seed,
                   mean_loss, per_sample: {stem: loss}}
  vis_data/eval/  the hooks' outputs (per-sample videos, loss in filename)

Masks are fixed by the config seed (same as train.py's val loss), so numbers
are comparable across checkpoints and with the val_loss in scalars.json.
"""

import argparse
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from custom import experiment as exp  # noqa: E402  (imports bootstrapper first)

import torch  # noqa: E402

from custom.checkpoints import load_mae_checkpoint  # noqa: E402
from custom.viz_hooks import ReconstructionVideoHook  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--dataset", default="", help="Kinetics data dir (required)")
    parser.add_argument("--checkpoint", default="",
                        help="'last' = the newest checkpoint of this config; a path or "
                             "name; omitted = the config's load_from")
    parser.add_argument("--mode", default="val", choices=["val", "pretrain", "test"],
                        help="Kinetics mode -> which csv is read (val.csv by default)")
    parser.add_argument("--batch-size", type=int, default=None)
    cli = parser.parse_args()

    if not cli.dataset.strip():
        print("Please inform a dataset.")
        raise SystemExit(1)
    dataset_dir = exp.bs.resolve(cli.dataset, exp.bs.DATASETS_ROOT)

    cfg = exp.load_config(cli.config)
    checkpoint, source = exp.resolve_checkpoint(
        cfg["work_dir"], cli.checkpoint, cfg.get("load_from"))
    if checkpoint is None:
        print("No --checkpoint given and the config has no load_from: "
              "there are no weights to evaluate.")
        raise SystemExit(1)
    ckpt_stem = osp.splitext(osp.basename(checkpoint))[0]
    ts = exp.timestamp()
    run_dir = osp.join(cfg["work_dir"], "compute_loss", f"{ts}-{ckpt_stem}")
    os.makedirs(osp.join(run_dir, "vis_data"), exist_ok=True)
    close_log = exp.install_tee(osp.join(run_dir, f"{ts}.log"))
    try:
        print(f"Using config:     {cfg['__config_path__']}")
        print(f"Using checkpoint: {checkpoint}  ({source})")
        print(f"Using dataset:    {dataset_dir} (mode {cli.mode})")

        seed = cfg["seed"]
        device = torch.device(cfg.get("device", "cuda"))
        batch_size = cli.batch_size or cfg.get("val_batch_size", 2)

        dataset_kwargs = dict(cfg["val_dataset"], path_to_data_dir=dataset_dir, mode=cli.mode)
        dataset = exp.build_dataset(dataset_kwargs)
        clips, stems = exp.load_fixed_clips(dataset, None, seed)
        print(f"Dataset: {len(clips)} clips from {len(dataset)} entries in {dataset_dir}")

        model = exp.build_model(cfg, device)
        load_mae_checkpoint(model, checkpoint)

        hooks = cfg["hooks"] or [ReconstructionVideoHook(mask_ratio=cfg["mask_ratio"])]
        per_sample = exp.run_hooks(hooks, model, clips, stems, "eval", run_dir,
                                   batch_size, device, seed)
        mean_loss = sum(per_sample.values()) / len(per_sample)

        print("\nPer-sample loss:")
        for stem in sorted(per_sample):
            print(f"  {stem}: {per_sample[stem]:.4f}")
        print(f"\nDataset size: {len(per_sample)}")
        print(f"Mean loss: {mean_loss:.4f}")

        with open(osp.join(run_dir, f"{ts}.json"), "w") as f:
            json.dump({
                "config": cfg["__config_path__"],
                "checkpoint": checkpoint,
                "checkpoint_source": source,
                "dataset": dataset_dir,
                "mode": cli.mode,
                "mask_ratio": cfg["mask_ratio"],
                "seed": seed,
                "mean_loss": mean_loss,
                "per_sample": dict(sorted(per_sample.items())),
            }, f, indent=2)
        print(f"Results written to: {run_dir}")
    finally:
        close_log()


if __name__ == "__main__":
    main()
