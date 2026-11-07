# mae_st_lcad: design record

What this fork changes over [facebookresearch/mae_st](https://github.com/facebookresearch/mae_st) (base commit `1e6b68a`), and why. Every change is one commit on `main`, and its message carries the detail and the checks that were run; this file is the overview. Claims were verified by running code; when one is overturned it is corrected here in place, with the date.

## 1. The domain

MAE-ST ([Feichtenhofer et al., 2022](https://arxiv.org/abs/2205.09113)) learns a video encoder by masking most spacetime patches of a clip and reconstructing them. Upstream trains it on Kinetics: natural-image video, roughly square frames, scaled and cropped freely as augmentation.

Here a frame is the **range image** of one sweep of a rotating lidar: row $i$ is laser beam $i$, column $j$ is azimuth step $j$, and the pixel is the measured range. For the Waymo Open Dataset top lidar that is 64 beams x 2650 azimuth steps per sweep, at 10 Hz ([Sun et al., 2020](https://arxiv.org/abs/1912.04838); the clips are produced by [lidar_sweep_viewer2](https://github.com/LCAD-UFES/lidar_sweep_viewer2)). Two properties follow, and they drive every domain change below:

- **The geometry is not square.** The frame is 40 times wider than tall, so the useful patch is too (4 beams x 50 azimuth steps at full resolution gives a 16 x 53 token grid per temporal slice).
- **The pixels are measurements.** Interpolating a range image invents beams and angles the sensor never measured, so the resize and resized-crop augmentations of natural images are wrong here.

## 2. Domain changes

| commit | change | why |
|---|---|---|
| `6d6ef89` | `models_mae.py` accepts `patch_size=(h, w)`; patchify, unpatchify and the decoder head use both sides; the decoder reads the token grid per axis | upstream asserted square frames and used one patch side everywhere |
| `82653f3` | crops take `(height, width)`; a crop larger than the frame raises | to cut the full 64 x 2650 frame or an azimuth sector; upstream silently returned a smaller tensor |
| `8ad368d` | `Kinetics(spatial_resample=False)` is the default: frames are decoded at native resolution and crops are cut, never rescaled; `rand_aug` with it raises | interpolation invents measurements (§1); `True` restores upstream |
| `12b4693` | `temporal_sample_index` and `spatial_sample_index` are constructor arguments (default -1, random, as upstream) | an overfit test must see the same clip and window every epoch |
| `8dcccd4` | the `rand_aug` path normalizes with the configured `mean`/`std` | range values have their own statistics; the hard-coded ImageNet values would silently override them |
| `3062b63` | `Kinetics(csv_file=...)` reads a csv with any name | one dataset folder can hold several slices (`train10.csv`, `train100.csv`, ...); `None` keeps upstream's `train/val/test.csv` |

With the default arguments and square sizes the model and the transforms are numerically identical to upstream (checked on fixed seeds, see the commits). The rectangular model and the data pipeline were also checked to be identical to the code of the September 2026 range-lidar overfit runs, which this fork supersedes.

## 3. Running on a current stack

Upstream no longer runs outside Meta's infrastructure on a current PyTorch. These commits fix that without changing behaviour:

| commit | change | why |
|---|---|---|
| `0fe58c3` | guards the Meta-internal `torch.fb` import; imports scikit-learn only where mAP needs it; `torch.load(weights_only=False)` for the repository's own checkpoints | import failures; torch >= 2.6 rejects the stored `argparse.Namespace` |
| `0ff8edf` | the video decoder uses PyAV instead of torchvision's private in-memory reader | torchvision 0.26 removed that API |
| `0ff8edf` | selective decoding counts frames from the stream's first timestamp | streams that do not start at pts 0 lost frames (one reference video returned 18 of 20) |
| `b96edb6` | `torch.optim.AdamW` instead of the private `_multi_tensor.AdamW` | removed from recent PyTorch |
| `daaab46` | attention uses `torch.nn.functional.scaled_dot_product_attention` instead of explicit q·kᵀ, softmax and ·v | the explicit form stores the whole attention matrix: at the 8480 tokens of a 64 x 2650 x 20 clip a ViT-B step took 3.3 s and 72 GB on a GB10, now 0.5 s and 4.9 GB (measured 2026-11-05); same weights, outputs equal to rounding |
| `d014b84` | `MaskedAutoencoderViT(attn_impl=...)`: `"sdpa"` (default) or `"explicit"`, the q·kᵀ form used before `daaab46` | `"explicit"` reproduces earlier runs bit for bit, at their cost (6.6x slower, 72 GB instead of 5 GB at 8480 tokens) |

## 4. Hooks for the tooling

The tooling in branch `mae_st_tooling` uses `main` as a library and does not edit it. Two small additions let it do so:

- `23c6eea`: `train_one_epoch(after_iter=None)` calls a callback after every step (visualizations every N iterations); `cuda.synchronize` is skipped on CPU.
- `56e1139`: `Kinetics` no longer prints its whole argument dict on construction.

## 5. Tests

`tests/` (stdlib `unittest`, CPU, a few seconds) covers the decoder, the rectangular model, the crops, the sampling positions, normalization, `csv_file` and the engine callback. Run from the clone, whatever its directory name:

```bash
python -m unittest discover -s tests -t .
```

## 6. Known open points

- **Run-to-run variance.** The backward of the fused kernel is not deterministic: two runs of one config agree to about 6 digits for the first epochs, then drift apart. The explicit form is deterministic, but it is one draw of the same spread, not a better one. With the overfit recipe (one step per epoch on the whole slice, lr 1e-3, fp16, no clipping) the training is unstable enough that the spread is large: three identical fused-attention runs of train10 stood at a train loss of 0.63, 0.71 and 0.73 after 1800 epochs, the explicit run at 0.63 (measured 2026-11-07). One run per configuration cannot rank configurations; compare several, or use a stabler recipe.
- **Deprecation warnings** from torch 2.11: `torch.cuda.amp.autocast` and `GradScaler` (`engine_pretrain.py`, `util/misc.py`).
