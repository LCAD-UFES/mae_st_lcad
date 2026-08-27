"""Checkpoint loading for upstream mae_st models.

The published MAE-ST checkpoints (e.g. video-mae-200x4-nonorm.pth) store the
attention projections FUSED as `attn.qkv.{weight,bias}` per block, while the
current upstream video_vit.Attention builds SEPARATE `attn.q/k/v` linears --
so a naive load_state_dict(strict=False) silently leaves every attention
layer at its random init (missing q/k/v, unexpected qkv; no shape mismatch,
hence no error). This was observed directly in the run_test smoke test.

split_fused_qkv() converts fused -> separate. The fused layout is the
standard timm one: rows [q; k; v] stacked along dim 0, so a 3-way chunk
recovers the three projections. This holds for encoder and decoder blocks
alike (both are video_vit.Block).
"""

import torch


def split_fused_qkv(state_dict):
    """Return a new state_dict with every `*.attn.qkv.*` entry split into
    `*.attn.q.*`, `*.attn.k.*`, `*.attn.v.*`. Non-qkv entries pass through."""
    out = {}
    for key, value in state_dict.items():
        if ".attn.qkv." in key:
            q, k, v = torch.chunk(value, 3, dim=0)
            out[key.replace(".attn.qkv.", ".attn.q.")] = q
            out[key.replace(".attn.qkv.", ".attn.k.")] = k
            out[key.replace(".attn.qkv.", ".attn.v.")] = v
        else:
            out[key] = value
    return out


def load_mae_checkpoint(model, path, device="cpu"):
    """Load a pretrain checkpoint into a models_mae model, converting the
    fused-qkv layout, and FAIL LOUDLY if anything but the known-benign keys
    is missing. Returns the checkpoint's stored args (or None)."""
    with open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state"))
    state_dict = split_fused_qkv(state_dict)

    msg = model.load_state_dict(state_dict, strict=False)
    # After the qkv split, a pretrain checkpoint must cover the entire MAE
    # model. Anything missing means a config/checkpoint mismatch and would
    # otherwise show up only as garbage reconstructions.
    if msg.missing_keys:
        raise RuntimeError(f"checkpoint {path} left keys uninitialized: "
                           f"{msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"[checkpoints] ignored unexpected keys: {msg.unexpected_keys}")
    print(f"[checkpoints] loaded {path}: all model keys matched")
    return checkpoint.get("args")
