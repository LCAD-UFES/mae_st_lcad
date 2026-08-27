"""Visualization hooks for MAE-ST reconstructions.

A hook is a callable with the signature

    hook(model, samples, stems, tag, run_dir) -> {stem: per_sample_loss}

where `model` is a models_mae.MaskedAutoencoderViT (already on its device,
eval mode), `samples` is a normalized [B, 3, T, H, W] batch on the same
device, `stems` names each sample (used in output filenames), `tag` labels
the moment ("eval", "epoch0042", ...) and `run_dir` is this run's output
directory. Hooks are instantiated in the config file and invoked by the
drivers (tools/train.py every `visualize_every_iterations`; tools/compute_loss.py
once), so adding a new visualization = writing a new callable here and listing
it in the config -- no driver changes. Outputs land in
run_dir/<out_subdir>/<tag>/.

ReconstructionVideoHook writes, per sample, a side-by-side video

    [ original | masked input | reconstruction | error map ]

as <stem>_loss<value>.mp4, with the PER-SAMPLE masked-region loss embedded in the filename
-- same convention as the sapiens overfit hook. The reconstruction panel is
the COMPOSITE the MAE literature shows: true pixels at visible positions,
prediction only at masked ones. The loss only ever penalizes masked patches,
so the raw decoder output at visible positions is untrained noise; the
composite is the honest rendering of what the model actually predicted.
"""

import os
import os.path as osp
from fractions import Fraction

import av
import numpy as np
import torch


class ReconstructionVideoHook:
    def __init__(
        self,
        mask_ratio=0.9,
        fps=7.5,
        mean=(0.45, 0.45, 0.45),
        std=(0.225, 0.225, 0.225),
        out_subdir="vis_data",
    ):
        self.mask_ratio = mask_ratio
        self.fps = fps
        self.mean = torch.tensor(mean).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1, 1)
        self.out_subdir = out_subdir

    @torch.no_grad()
    def __call__(self, model, samples, stems, tag, run_dir):
        _, pred, mask = model(samples, mask_ratio=self.mask_ratio)
        mask = mask.unsqueeze(-1)  # [B, L, 1]; 1 = masked/removed

        # The loss target subsamples pred_t_dim frames from the input (all of
        # them when pred_t_dim == num_frames); mirror forward_loss exactly so
        # every panel lives on the same frames as the prediction.
        frame_idx = (
            torch.linspace(0, samples.shape[2] - 1, model.pred_t_dim)
            .long()
            .to(samples.device)
        )
        ref = torch.index_select(samples, 2, frame_idx)
        target = model.patchify(ref)  # [B, L, u*p*p*3]; also sets patch_info

        # forward_loss compares pred against the (optionally per-patch
        # normalized) target; reproduce that space for the per-sample loss,
        # then bring pred back to dataset-normalized space for rendering.
        if model.norm_pix_loss:
            t_mean = target.mean(dim=-1, keepdim=True)
            t_var = target.var(dim=-1, keepdim=True)
            target_loss_space = (target - t_mean) / (t_var + 1.0e-6) ** 0.5
            pred_render = pred * (t_var + 1.0e-6) ** 0.5 + t_mean
        else:
            target_loss_space = target
            pred_render = pred

        per_patch_mse = ((pred - target_loss_space) ** 2).mean(dim=-1, keepdim=True)
        per_sample_loss = (per_patch_mse * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))

        # Composite (paper convention) and masked-input panels, in patch
        # space; 0 is the dataset-normalized gray (unnormalizes to ~0.45*255).
        composite = model.unpatchify(target * (1 - mask) + pred_render * mask)
        masked_in = model.unpatchify(target * (1 - mask))

        ref_u8 = self._to_uint8(ref)
        masked_u8 = self._to_uint8(masked_in)
        composite_u8 = self._to_uint8(composite)

        out_dir = osp.join(run_dir, self.out_subdir, tag)
        os.makedirs(out_dir, exist_ok=True)
        losses = {}
        for i, stem in enumerate(stems):
            diff = np.abs(
                ref_u8[i].astype(np.int16) - composite_u8[i].astype(np.int16)
            ).mean(axis=-1).astype(np.uint8)
            diff_u8 = np.repeat(diff[..., None], 3, axis=-1)

            panels = np.concatenate(
                [ref_u8[i], masked_u8[i], composite_u8[i], diff_u8], axis=2
            )  # [T, H, 4W, 3]
            base = f"{stem}_loss{float(per_sample_loss[i]):.4f}"
            write_video(osp.join(out_dir, base + ".mp4"), panels, self.fps)
            losses[stem] = float(per_sample_loss[i])
        return losses

    def _to_uint8(self, video):
        """dataset-normalized [B, 3, T, H, W] -> uint8 [B, T, H, W, 3]"""
        video = video.cpu() * self.std + self.mean  # undo tensor_normalize
        video = (video * 255.0).clamp(0, 255).to(torch.uint8)
        return video.permute(0, 2, 3, 4, 1).numpy()


def write_video(path, frames, fps):
    """Encode uint8 [T, H, W, 3] frames as h264 mp4 (no ffmpeg CLI on this
    machine; PyAV's bundled libx264 does the job)."""
    height, width = frames.shape[1:3]
    rate = Fraction(fps).limit_denominator(1000)
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for n, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = n
            frame.time_base = Fraction(1, 1) / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())  # flush
