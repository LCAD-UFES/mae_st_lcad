#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for Masked Autoencoders As Spatiotemporal Learners (MAE-ST).
This script loads a pre-trained MAE-ST checkpoint, processes an input video (or directory of videos),
applies spatiotemporal masking, reconstructs the masked patches, and saves a side-by-side
visualization of (Original Video | Masked Video | Reconstructed Video).
"""

import os
import sys
import types
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

from functools import partial

# Ensure parent directory is in python path to import models_mae and utility modules
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
    parser = argparse.ArgumentParser(description="MAE-ST Video Reconstruction Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="mae_vit_huge_patch16",
        choices=["mae_vit_base_patch16", "mae_vit_large_patch16", "mae_vit_huge_patch14", "mae_vit_huge_patch16"],
        help="Model architecture type to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/k700_400ep_rep4_mr9_16x4_huge.pyth",
        help="Path to the pre-trained MAE-ST model checkpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./ActivityNet/Crawler/Kinetics/dataset/test/-7pWlHwSRbU_000117_000127.mp4",
        help="Path to a single video file (.mp4) or a directory containing videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_visualizations",
        help="Directory to save the reconstructed output videos"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.95,
        help="Masking ratio (percentage of removed patches, e.g. 0.90 for 90%)"
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
        "--fps",
        type=float,
        default=2.0,
        help="Frame rate of the output visualization video"
    )
    parser.add_argument(
        "--t_patch_size",
        type=int,
        default=2,
        help="Temporal patch size of the model"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Spatial patch size of the model"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of input frames to sample from the video"
    )
    parser.add_argument(
        "--pred_t_dim",
        type=int,
        default=8,
        help="Number of predicted frames (temporal dimension of prediction)"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="whole_video",
        choices=["single_clip", "whole_video"],
        help="Inference mode: single_clip (center 16 frames) or whole_video (chunk-by-chunk over all frames)"
    )
    parser.add_argument(
        "--deblock",
        action="store_true",
        help="Apply post-processing bilateral filter to smooth out block/grid artifacts in reconstructed patches"
    )
    return parser.parse_args()


def custom_forward_decoder(self, x, ids_restore):
    """
    Custom forward decoder method that handles the flat decoder_pos_embed 
    of the pre-trained Kinetics checkpoint.
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
    Adapts the state_dict from the pre-trained Facebook/Kinetics checkpoint
    to match the class parameter names in models_mae.py.
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
    Instantiates the MAE-ST model and loads the pre-trained weights.
    """
    print(f"[*] Initializing MAE-ST model ({args.model})...")
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
        t_patch_size=args.t_patch_size,
        decoder_depth=4,
        pred_t_dim=args.pred_t_dim,
        mask_type="st",
        norm_pix_loss=args.norm_pix_loss
    )
    
    # Compute decoder position embedding shape dynamically
    num_patches = model.patch_embed.num_patches
    decoder_embed_dim = model.decoder_embed.out_features
    total_pos_tokens = num_patches + 1
    
    model.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_pos_tokens, decoder_embed_dim))
    
    # Replace default forward_decoder with custom flat positional embedding handler
    model.forward_decoder = types.MethodType(custom_forward_decoder, model)
    
    print(f"[*] Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("model", ckpt))
    
    adapted_dict = adapt_state_dict(state_dict)
    msg = model.load_state_dict(adapted_dict, strict=False)
    print(f"[+] Loaded successfully! Missing keys (expected): {msg.missing_keys}")
    
    model.to(args.device)
    model.eval()
    return model


def load_and_preprocess_video(video_path, args):
    """
    Loads an input video using OpenCV, resizes and crops all frames to 224x224,
    and returns either a single clip of 16 frames or a list of 16-frame chunks
    covering the entire video.
    """
    print(f"[*] Loading and preprocessing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    frames = []
    # Get original fps
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    T = len(frames)
    if T == 0:
        raise ValueError(f"Video file is empty (0 frames): {video_path}")
        
    # Convert frames list to a single torch tensor
    tensor = torch.from_numpy(np.stack(frames)).float()  # (T, H, W, C)
    
    # Resize and crop to 224x224
    _, H, W, C = tensor.shape
    if H < W:
        new_h = 224
        new_w = int(W * (224 / H))
    else:
        new_w = 224
        new_h = int(H * (224 / W))
        
    tensor = tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    
    y_start = (new_h - 224) // 2
    x_start = (new_w - 224) // 2
    tensor = tensor[:, :, y_start:y_start+224, x_start:x_start+224].permute(0, 2, 3, 1)  # (T, H, W, C)
    
    if args.inference_mode == "single_clip":
        # Frame sampling: we need exactly num_frames (16) frames
        # If the video is long enough (>= num_frames * stride), extract center clip
        stride = 4
        clip_sz = args.num_frames * stride
        if T >= clip_sz:
            start_idx = (T - clip_sz) // 2
            indices = np.arange(start_idx, start_idx + clip_sz, stride)[:args.num_frames]
        else:
            indices = np.linspace(0, T - 1, args.num_frames, dtype=int)
            
        sampled_frames = tensor[indices]  # (16, 224, 224, C)
        sampled_frames = sampled_frames / 255.0
        
        # Normalize
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 1, 3)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 1, 3)
        normalized = (sampled_frames - mean) / std
        
        # Make a list containing 1 chunk
        chunks = [normalized.permute(3, 0, 1, 2)]  # [(C, 16, 224, 224)]
    else:
        # Whole video mode: split into consecutive chunks of 16 frames
        num_chunks = int(np.ceil(T / args.num_frames))
        pad_len = num_chunks * args.num_frames - T
        if pad_len > 0:
            last_frame = tensor[-1:]
            padded_tensor = torch.cat([tensor, last_frame.repeat(pad_len, 1, 1, 1)], dim=0)
        else:
            padded_tensor = tensor
            
        chunks = []
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 1, 3)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 1, 3)
        
        for i in range(num_chunks):
            start_idx = i * args.num_frames
            chunk_frames = padded_tensor[start_idx : start_idx + args.num_frames] / 255.0
            normalized = (chunk_frames - mean) / std
            chunks.append(normalized.permute(3, 0, 1, 2))  # (C, 16, 224, 224)
            
    return chunks, orig_fps


def run_reconstruction(model, chunks, args):
    """
    Runs the forward pass on the model for each 16-frame chunk, applies masking, 
    reconstructs the video, and returns the concatenated (Original, Masked, Reconstructed) frame tensors.
    """
    orig_list, masked_list, recon_list = [], [], []
    
    mean_val = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).to(args.device)
    std_val = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).to(args.device)
    
    print(f"[*] Running MAE-ST reconstruction over {len(chunks)} chunk(s) (Mask ratio: {args.mask_ratio * 100}%)...")
    
    for idx, chunk in enumerate(chunks):
        input_tensor = chunk.unsqueeze(0).to(args.device)  # (1, C, 16, 224, 224)
        
        with torch.no_grad():
            # Pass index=1 as required by models_mae.py forward
            loss, pred, mask, comparison = model(input_tensor, index=1, mask_ratio=args.mask_ratio, visualize=True)
            
        orig = comparison[0, 0] # (3, T, H, W)
        masked = comparison[0, 1] # (3, T, H, W)
        recon = comparison[0, 2] # (3, T, H, W)
        
        # Denormalize to [0, 1]
        orig_vis = torch.clamp(orig * std_val + mean_val, 0, 1)
        masked_vis = torch.clamp(masked * std_val + mean_val, 0, 1)
        recon_vis = torch.clamp(recon * std_val + mean_val, 0, 1)
        
        orig_list.append(orig_vis.cpu())
        masked_list.append(masked_vis.cpu())
        recon_list.append(recon_vis.cpu())
        
    # Concatenate all chunks along the temporal dimension (axis 1)
    orig_all = torch.cat(orig_list, dim=1)  # (3, total_frames, 224, 224)
    masked_all = torch.cat(masked_list, dim=1)  # (3, total_frames, 224, 224)
    recon_all = torch.cat(recon_list, dim=1)  # (3, total_frames, 224, 224)
    
    return orig_all, masked_all, recon_all


def save_visualization(orig, masked, recon, output_path, fps, args):
    """
    Creates a side-by-side comparison video (Original | Masked | Reconstructed) and saves it as .mp4.
    """
    print(f"[*] Saving visualization to: {output_path} with {fps:.2f} FPS")
    # Convert PyTorch tensors to numpy arrays: shape (3, T, H, W) -> (T, H, W, 3)
    orig_np = orig.permute(1, 2, 3, 0).cpu().numpy()
    masked_np = masked.permute(1, 2, 3, 0).cpu().numpy()
    recon_np = recon.permute(1, 2, 3, 0).cpu().numpy()
    
    num_frames = orig_np.shape[0]
    height, width, _ = orig_np.shape[1:]
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))
    
    for t in range(num_frames):
        f_orig = (orig_np[t] * 255.0).astype(np.uint8)
        f_masked = (masked_np[t] * 255.0).astype(np.uint8)
        f_recon = (recon_np[t] * 255.0).astype(np.uint8)
        
        if args.deblock:
            # Apply a light bilateral filter to smooth out block artifacts
            f_recon = cv2.bilateralFilter(f_recon, d=9, sigmaColor=75, sigmaSpace=75)
            
        # Concatenate horizontally
        combined = np.concatenate([f_orig, f_masked, f_recon], axis=1)
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
        
    out.release()
    print(f"[+] Output saved: {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine input files
    if os.path.isdir(args.input):
        print(f"[*] Input is a directory. Scanning for videos in: {args.input}")
        video_files = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(root, file))
        print(f"[+] Found {len(video_files)} video files.")
    else:
        video_files = [args.input]
        
    if len(video_files) == 0:
        print("[!] No video files found to process.")
        return
        
    # Get model
    model = get_model(args)
    
    # Process each video
    for video_path in video_files:
        try:
            chunks, orig_fps = load_and_preprocess_video(video_path, args)
            orig, masked, recon = run_reconstruction(model, chunks, args)
            
            # Determine output FPS: if whole_video, match the original FPS exactly
            if args.inference_mode == "whole_video":
                out_fps = orig_fps
            else:
                out_fps = args.fps
                
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"{video_name}_mr{int(args.mask_ratio*100)}_reconstructed.mp4"
            output_path = os.path.join(args.output_dir, output_name)
            
            save_visualization(orig, masked, recon, output_path, out_fps, args)
        except Exception as e:
            print(f"[!] Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n[+] Inference and visualization generation completed successfully!")


if __name__ == "__main__":
    main()
