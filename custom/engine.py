"""Training step for one epoch -- a copy of upstream engine_pretrain.py.

This is ~/mae_st/engine_pretrain.py::train_one_epoch reproduced VERBATIM
(same meters, same lr schedule call, same NaN guard, same tensorboard
scalars) with exactly ONE addition: the `after_iter` callback, invoked at
the end of every optimizer step with the 1-based global iteration number
(epoch * len(data_loader) + step + 1). tools/train.py uses it to run the
visualization hooks every `visualize_every_iterations` -- a hook point the
upstream engine does not offer, and which cannot be added without editing
the pristine clone. Whenever upstream changes, re-diff this file against
engine_pretrain.py.

Callback contract: `after_iter(global_iter)`; it may switch the model to
eval mode, but must leave it in train mode when it returns (the engine does
not re-assert it inside the loop).
"""

import math
from typing import Iterable

import mae_st.util.lr_sched as lr_sched
import mae_st.util.misc as misc
import torch
from iopath.common.file_io import g_pathmgr as pathmgr


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    fp32=False,
    after_iter=None,  # <-- the one addition
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        if len(samples.shape) == 6:
            b, r, c, t, h, w = samples.shape
            samples = samples.reshape(b * r, c, t, h, w)

        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, _, _ = model(
                samples,
                mask_ratio=args.mask_ratio,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            for _ in range(args.num_checkpoint_del):
                try:
                    path = misc.get_last_checkpoint(args)
                    pathmgr.rm(path)
                    print(f"remove checkpoint {path}")
                except Exception as _:
                    pass
            raise Exception("Loss is {}, stopping training".format(loss_value))

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=args.clip_grad,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
        metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
        metric_logger.update(gpu_mem=misc.gpu_mem_usage())
        metric_logger.update(mask_ratio=args.mask_ratio)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000 * args.repeat_aug
            )
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

        # --- addition: per-iteration hook point ---
        if after_iter is not None:
            after_iter(epoch * len(data_loader) + data_iter_step + 1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
