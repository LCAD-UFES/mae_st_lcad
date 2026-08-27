"""Train an MAE-ST model with our own epoch loop around the upstream engine.

    python tools/train.py <config> [--resume] [--tensorboard on|off]

<config> is a path, or a bare file name looked up in EXPERIMENTS_ROOT/configs/
(see bootstrapper.py for where that is).

Why our own loop: upstream main_pretrain.py has no validation, no hook
points and a private optimizer API that no longer exists. Here the upstream
pieces (Kinetics, models_mae, the per-epoch training step, NativeScaler,
lr_sched) are used as a library and the loop adds a deterministic validation
loss every epoch, checkpoints every `save_every_epochs`, visualization hooks
every `visualize_every_iterations`, a scalars.json for plot_loss.py, and
resume.

Epoch numbering: the engine receives the upstream 0-based epoch index;
everything user-facing (scalars.json, epoch_N.pth, epochXXXX tags) counts
epochs COMPLETED, 1-based, like mmengine.

--resume restores model/optimizer/scaler/epoch from work_dir/last_checkpoint,
starts a NEW <ts>/ run dir (own log, config copy, tensorboard) and seeds its
scalars.json with the previous run's records so the curve stays continuous.
"""

import argparse
import os.path as osp
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from custom import experiment as exp  # noqa: E402  (imports bootstrapper first)

import torch  # noqa: E402
import torch.backends.cudnn as cudnn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import mae_st.util.misc as misc  # noqa: E402
from custom.checkpoints import load_mae_checkpoint  # noqa: E402
from custom.engine import train_one_epoch  # noqa: E402


def build_engine_args(cfg, work_dir):
    """The argparse.Namespace the upstream engine / lr_sched / misc read."""
    eff_batch = cfg["batch_size"] * cfg.get("repeat_aug", 1) * cfg.get("accum_iter", 1)
    lr = cfg.get("lr")
    if lr is None:
        lr = cfg["blr"] * eff_batch / 256  # upstream linear scaling rule
    return argparse.Namespace(
        # read by custom.engine.train_one_epoch
        accum_iter=cfg.get("accum_iter", 1),
        mask_ratio=cfg["mask_ratio"],
        clip_grad=cfg.get("clip_grad"),
        repeat_aug=cfg.get("repeat_aug", 1),
        # 0 is load-bearing: on a non-finite loss the engine deletes
        # `num_checkpoint_del` files chosen by misc.get_last_checkpoint, whose
        # filter is `"checkpoint" in name` -- which matches our
        # `last_checkpoint` sentinel. With 0 it only raises.
        num_checkpoint_del=0,
        output_dir=work_dir,
        # read by mae_st.util.lr_sched.adjust_learning_rate
        lr=lr,
        min_lr=cfg["min_lr"],
        warmup_epochs=cfg["warmup_epochs"],
        epochs=cfg["max_epochs"],
        # provenance only (stored in the checkpoint)
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        fp32=cfg.get("fp32", True),
        weight_decay=cfg.get("weight_decay", 0.05),
        model=cfg["model_name"],
        config=cfg["__config_path__"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="config path, or a name in EXPERIMENTS_ROOT/configs/")
    parser.add_argument("--resume", action="store_true",
                        help="continue from work_dir/last_checkpoint")
    parser.add_argument("--tensorboard", choices=["on", "off"], default="on",
                        help="write TensorBoard events to <run_dir>/tb (default on; "
                             "cheap, and it is what lets a dashboard be attached later)")
    cli = parser.parse_args()

    cfg = exp.load_config(cli.config)
    work_dir = cfg["work_dir"]
    ts = exp.timestamp()
    run_dir = exp.make_run_dir(work_dir, cfg["__config_path__"], ts)
    close_log = exp.install_tee(osp.join(run_dir, f"{ts}.log"))
    log_writer = None
    try:
        print(f"[train] config:  {cfg['__config_path__']}")
        print(f"[train] run_dir: {run_dir}")
        print(f"[train] resume:  {cli.resume}")

        seed = cfg["seed"]
        exp.seed_everything(seed)
        cudnn.benchmark = True
        device = torch.device(cfg.get("device", "cuda"))
        args = build_engine_args(cfg, work_dir)
        print("[train] engine args:", vars(args))

        # --- data ------------------------------------------------------------
        train_ds = exp.build_dataset(cfg["train_dataset"])
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            sampler=torch.utils.data.RandomSampler(train_ds),
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,  # upstream uses True; with 10 videos that drops data
        )
        val_ds = exp.build_dataset(cfg["val_dataset"])
        val_clips, val_stems = exp.load_fixed_clips(val_ds, cfg.get("val_max_videos"), seed)
        val_batch = cfg.get("val_batch_size", cfg["batch_size"])
        print(f"[train] {len(train_ds)} train videos, {len(train_loader)} iters/epoch; "
              f"{len(val_clips)} fixed val clips")

        # --- model / optimizer ----------------------------------------------
        model = exp.build_model(cfg, device)
        if not cli.resume and cfg.get("load_from"):
            load_mae_checkpoint(model, exp.bs.resolve(cfg["load_from"], exp.bs.CHECKPOINTS_DIR))

        param_groups = misc.add_weight_decay(
            model, cfg.get("weight_decay", 0.05), bias_wd=cfg.get("bias_wd", False))
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                      betas=tuple(cfg.get("betas", (0.9, 0.95))))
        loss_scaler = misc.NativeScalerWithGradNormCount(fp32=args.fp32)
        print(f"[train] lr {args.lr:.2e} (min {args.min_lr:.2e}, warmup {args.warmup_epochs})")

        # --- resume ------------------------------------------------------------
        start_epoch = 0
        if cli.resume:
            path = exp.read_last_checkpoint(work_dir)
            if path is None:
                raise FileNotFoundError(f"--resume given but no last_checkpoint in {work_dir}")
            start_epoch, ckpt = exp.load_resume_checkpoint(path, model, optimizer, loss_scaler)
            prev = ckpt.get("run_dir")
            prev_scalars = osp.join(prev, "vis_data", "scalars.json") if prev else None
            if prev_scalars and osp.isfile(prev_scalars):
                carried = [r for r in exp.read_scalars(prev_scalars) if r["epoch"] <= start_epoch]
                for r in carried:
                    exp.append_scalar(run_dir, r)
                print(f"[train] carried {len(carried)} scalar records from {prev_scalars}")

        if cli.tensorboard == "on":
            # flush_secs=10 (default 120) so a live dashboard lags seconds, not minutes
            log_writer = SummaryWriter(log_dir=osp.join(run_dir, "tb"), flush_secs=10)
            print(f"[train] tensorboard events -> {log_writer.log_dir}")
        else:
            print("[train] tensorboard off (scalars.json is still written)")

        # --- cadence -----------------------------------------------------------
        max_epochs = cfg["max_epochs"]
        save_every = cfg.get("save_every_epochs", 0)
        vis_every = cfg.get("visualize_every_iterations", 0)
        iters_per_epoch = len(train_loader)

        def after_iter(global_iter):
            if vis_every > 0 and global_iter % vis_every == 0:
                epochs_done = (global_iter - 1) // iters_per_epoch + 1
                tag = f"epoch{epochs_done:04d}_iter{global_iter:06d}"
                print(f"[train] visualizing {tag}")
                exp.run_hooks(cfg["hooks"], model, val_clips, val_stems, tag, run_dir,
                              val_batch, device, seed)
                model.train(True)  # hooks leave the model in eval mode

        # --- epoch loop --------------------------------------------------------
        print(f"[train] epochs {start_epoch + 1}..{max_epochs}")
        t0 = time.time()
        for epoch in range(start_epoch, max_epochs):
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, args=args, fp32=args.fp32, after_iter=after_iter)
            val_loss = exp.masked_loss(model, val_clips, cfg["mask_ratio"], val_batch, device, seed)
            epochs_done = epoch + 1

            if log_writer is not None:
                log_writer.add_scalar("val_loss", val_loss, epochs_done)
            record = {"epoch": epochs_done, "loss": train_stats["loss"],
                      "val_loss": val_loss, "lr": train_stats["lr"], "time": time.time()}
            exp.append_scalar(run_dir, record)
            print(f"[train] epoch {epochs_done}/{max_epochs}  loss {record['loss']:.4f}  "
                  f"val_loss {val_loss:.4f}  lr {record['lr']:.2e}  "
                  f"elapsed {time.time() - t0:.0f}s")

            last = epochs_done == max_epochs
            if last or (save_every > 0 and epochs_done % save_every == 0):
                exp.save_checkpoint(work_dir, epochs_done, model, optimizer, loss_scaler,
                                    args, cfg["__config_path__"], run_dir,
                                    cfg.get("max_keep_ckpts", 2))
        print(f"[train] done in {time.time() - t0:.0f}s -> {run_dir}")
    finally:
        if log_writer is not None:
            log_writer.close()
        close_log()


if __name__ == "__main__":
    main()
