#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for Masked Autoencoders As Spatiotemporal Learners (MAE-ST).
This script is based on the 'video_mae_visualize.ipynb' demo. It loads a video,
applies spatiotemporal masking, reconstructs it using the pre-trained MAE-ST model,
and generates a high-resolution matplotlib grid (Original | Masked | Reconstructed)
saved as an image file.
"""

import os
import sys
import types
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from functools import partial

# Ensure parent directory is in python path to import models_mae
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14, MaskedAutoencoderViT

def mae_vit_huge_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="MAE-ST Video Grid Visualization")
    parser.add_argument(
        "--model",
        type=str,
        default="mae_vit_large_patch16",
        choices=["mae_vit_base_patch16", "mae_vit_large_patch16", "mae_vit_huge_patch14", "mae_vit_huge_patch16"],
        help="Model architecture type to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/k400_400ep_rep4_mr9_16x4.pyth",
        help="Path to the pre-trained MAE-ST model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./ActivityNet/Crawler/Kinetics/dataset/test/-7pWlHwSRbU_000117_000127.mp4",
        help="Path to the input video file (.mp4)"
    )
    parser.add_argument(
        "--output_img",
        type=str,
        default="./output_visualizations/reconstruction_grid.png",
        help="Path to save the output grid visualization image"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.90,
        help="Masking ratio (percentage of removed patches)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--norm_pix_loss",
        type=bool,
        default=True,
        help="Whether the model was trained with per-patch normalized pixel loss"
    )
    parser.add_argument(
        "--deblock",
        action="store_true",
        help="Apply post-processing bilateral filter to smooth out block/grid artifacts in reconstructed patches"
    )
    return parser.parse_args()


def custom_forward_decoder(self, x, ids_restore):
    """
    Custom forward decoder method to handle flat decoder_pos_embed.
    """
    N = x.shape[0]
    T = self.patch_embed.t_grid_size
    H = W = self.patch_embed.grid_size
    x = self.decoder_embed(x)
    C = x.shape[-1]
    
    # Append mask tokens to sequence
    mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
    x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)
    if self.mask_type == "st":
        x_ = x_.view([N, T * H * W, C])
    elif self.mask_type == "t":
        x_ = x_.view([N * T, H * W, C])
    elif self.mask_type == "tube":
        x_ = x_.reshape([N, T, H * W, C])
        x_ = torch.einsum("ntlc->nltc", x_)
        x_ = x_.reshape([N, H * W, T * C])
    else:
        raise NotImplementedError(f"not supported mask type {self.mask_type}")
        
    x_ = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
    )  # Unshuffle
    
    if self.mask_type in ["st", "t"]:
        x = x_.view([N, T * H * W, C])
    elif self.mask_type == "tube":
        x = x_.reshape([N, H * W, T, C])
        x = torch.einsum("nltc->ntlc", x)
        x = x.reshape([N, T * H * W, C])
    else:
        raise NotImplementedError(f"not supported mask type {self.mask_type}")
        
    decoder_pos_embed = self.decoder_pos_embed[:, 1:, :]
    x = x + decoder_pos_embed
    
    attn = self.decoder_blocks[0].attn
    requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
    if requires_t_shape:
        x = x.view([N, T, H * W, C])
        
    # Apply Transformer blocks
    for blk in self.decoder_blocks:
        x = blk(x)
    x = self.decoder_norm(x)
    
    # Predictor projection
    x = self.decoder_pred(x)
    
    if requires_t_shape:
        x = x.view([N, T * H * W, -1])
        
    x = x[:, :, :]
    return x


def adapt_state_dict(state_dict):
    """
    Adapts PySlowFast keys to models_mae.py class names.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("pred_head.transforms.0.4."):
            new_key = k.replace("pred_head.transforms.0.4.", "decoder_norm.")
        elif k.startswith("pred_head.projections.0."):
            new_key = k.replace("pred_head.projections.0.", "decoder_pred.")
        elif k.startswith("pred_head.transforms.0."):
            parts = k.split(".")
            block_idx = parts[3]
            suffix = ".".join(parts[4:])
            new_key = f"decoder_blocks.{block_idx}.{suffix}"
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def get_model(args):
    """
    Loads and adapts the pre-trained MAE-ST model.
    """
    print(f"[*] Initializing model ({args.model})...")
    if args.model == "mae_vit_base_patch16":
        model_fn = mae_vit_base_patch16
    elif args.model == "mae_vit_large_patch16":
        model_fn = mae_vit_large_patch16
    elif args.model == "mae_vit_huge_patch14":
        model_fn = mae_vit_huge_patch14
    elif args.model == "mae_vit_huge_patch16":
        model_fn = mae_vit_huge_patch16
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model_fn(
        sep_pos_embed=True,
        t_patch_size=2,
        decoder_depth=4,
        pred_t_dim=8,
        mask_type="st",
        norm_pix_loss=args.norm_pix_loss
    )
    
    # Compute decoder position embedding shape dynamically
    num_patches = model.patch_embed.num_patches
    decoder_embed_dim = model.decoder_embed.out_features
    total_pos_tokens = num_patches + 1
    
    model.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_pos_tokens, decoder_embed_dim))
    model.forward_decoder = types.MethodType(custom_forward_decoder, model)
    
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
    adapted_dict = adapt_state_dict(state_dict)
    model.load_state_dict(adapted_dict, strict=False)
    
    model.to(args.device)
    model.eval()
    return model


def load_and_preprocess_video(video_path):
    """
    Loads input video and prepares it.
    """
    print(f"[*] Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Video file is empty (0 frames): {video_path}")
        
    tensor = torch.from_numpy(np.stack(frames)).float()
    T, H, W, C = tensor.shape
    
    # Resize and Crop to 224
    if H < W:
        new_h = 224
        new_w = int(W * (224 / H))
    else:
        new_w = 224
        new_h = int(H * (224 / W))
        
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    
    y_start = (new_h - 224) // 2
    x_start = (new_w - 224) // 2
    tensor = tensor[:, :, y_start:y_start+224, x_start:x_start+224].permute(0, 2, 3, 1)
    
    # Sample 16 frames (center clip)
    stride = 4
    clip_sz = 16 * stride
    if T >= clip_sz:
        start_idx = (T - clip_sz) // 2
        indices = np.arange(start_idx, start_idx + clip_sz, stride)[:16]
    else:
        indices = np.linspace(0, T - 1, 16, dtype=int)
        
    sampled_frames = tensor[indices] / 255.0
    
    # Normalize
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 1, 3)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 1, 3)
    normalized = (sampled_frames - mean) / std
    
    return normalized.permute(3, 0, 1, 2).unsqueeze(0)


def run_mae(model, input_tensor, args):
    """
    Performs forward pass, masking, and returns denormalized visual tensors.
    """
    print(f"[*] Running MAE reconstruction...")
    input_tensor = input_tensor.to(args.device)
    
    with torch.no_grad():
        # Pass index=1 as required by models_mae.py forward
        loss, pred, mask, comparison = model(input_tensor, index=1, mask_ratio=args.mask_ratio, visualize=True)
        
    orig = comparison[0, 0] # (3, T, H, W)
    masked = comparison[0, 1] # (3, T, H, W)
    recon = comparison[0, 2] # (3, T, H, W)
    
    # Sub-sample original video to get 8 frames for clean grid plotting
    indices = torch.linspace(0, orig.shape[1] - 1, 8).long().to(args.device)
    orig = torch.index_select(orig, 1, indices)
    masked = torch.index_select(masked, 1, indices)
    recon = torch.index_select(recon, 1, indices)
    
    # Denormalize to RGB [0, 1]
    mean_val = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).to(args.device)
    std_val = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).to(args.device)
    
    orig_vis = torch.clamp(orig * std_val + mean_val, 0, 1)
    masked_vis = torch.clamp(masked * std_val + mean_val, 0, 1)
    recon_vis = torch.clamp(recon * std_val + mean_val, 0, 1)
    
    return orig_vis, masked_vis, recon_vis


def plot_and_save_grid(orig, masked, recon, output_img_path, args):
    """
    Creates a 3x8 frame comparison plot (Original, Masked, Reconstructed)
    exactly mirroring 'plot_input' from video_mae_visualize.ipynb.
    """
    print(f"[*] Plotting grid and saving to: {output_img_path}")
    
    # Permute tensors: (3, T, H, W) -> (T, H, W, 3)
    orig_np = orig.permute(1, 2, 3, 0).cpu().numpy()
    masked_np = masked.permute(1, 2, 3, 0).cpu().numpy()
    recon_np = recon.permute(1, 2, 3, 0).cpu().numpy()
    
    num_cols = orig_np.shape[0]  # T = 8
    
    # Set up matplotlib figure (nrows=3, ncols=8)
    fig, axes = plt.subplots(nrows=3, ncols=num_cols, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    row_labels = ["Original Video", f"Masked Video", "Reconstructed Video"]
    
    for r in range(3):
        for c in range(num_cols):
            ax = axes[r, c]
            ax.axis("off")
            
            if r == 0:
                img = orig_np[c]
            elif r == 1:
                img = masked_np[c]
            else:
                img = recon_np[c]
                if args.deblock:
                    # Scale to uint8, apply bilateral, convert back to float
                    f_recon_uint8 = (img * 255.0).astype(np.uint8)
                    f_recon_filtered = cv2.bilateralFilter(f_recon_uint8, d=9, sigmaColor=75, sigmaSpace=75)
                    img = f_recon_filtered.astype(np.float32) / 255.0
                
            ax.imshow(img)
            
            # Label the first image of each row
            if c == 0:
                ax.text(
                    -15, 112, row_labels[r], 
                    rotation=90, va="center", ha="right", 
                    fontsize=16, weight="bold", color="black"
                )
                
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_img_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[+] Successfully saved grid plot to: {output_img_path}")


def main():
    args = parse_args()
    
    # 1. Load pre-trained MAE-ST model
    model = get_model(args)
    
    # 2. Load and preprocess input video
    input_tensor = load_and_preprocess_video(args.input)
    
    # 3. Perform masking and reconstruction
    orig, masked, recon = run_mae(model, input_tensor, args)
    
    # 4. Generate the 3x8 frame grid plot and save to disk
    plot_and_save_grid(orig, masked, recon, args.output_img, args)
    
    print("\n[+] Grid visualization generated successfully!")


if __name__ == "__main__":
    main()
