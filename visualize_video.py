import sys
import os
import argparse
import torch
import numpy as np
import cv2
import av

# Append necessary paths
sys.path.append('..')
sys.path.append('.')
sys.path.append('./slowfast')

from mae.models_mae import mae_vit_large_patch16
from util.decoder.utils import tensor_normalize, spatial_sampling

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(description="MAE Video Inference and Visualization")
    parser.add_argument(
        "--video_path", 
        type=str, 
        default="demo/qZ_lFjCiR1c_000104_000114.avi",
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output_reconstruction.mp4",
        help="Path to save the output video visualization"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str,
        default="checkpoints/video-mae-200x4-nonorm.pth",
        # default="checkpoints/video-mae-100x4-joint.pth",
        help="Path to the pre-trained MAE checkpoint"
    )
    parser.add_argument(
        "--mask_ratio", 
        type=float, 
        default=0.95,
        help="Masking ratio (default: 0.9)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)"
    )
    return parser.parse_args()

def load_video_frames(video_path):
    """
    Loads all frames from the video using OpenCV.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        
    frames = []
    r = True
    while r:
        r, f = cap.read()
        if r:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames could be read from the video: {video_path}")
        
    print(f"Loaded {len(frames)} frames from {video_path} at {fps:.2f} FPS.")
    return torch.from_numpy(np.stack(frames)), fps

def process_video_mae(frames, model, mask_ratio, device):
    """
    Processes the video frames chunk by chunk using MAE.
    """
    T_total = frames.shape[1]  # Shape is [3, T_total, H, W] after normalize and permute
    
    all_originals = []
    all_masked = []
    all_reconstructs = []
    
    model.to(device)
    model.eval()
    
    # Process the entire video sequentially in chunks of 16 frames
    for i in range(0, T_total, 16):
        if i + 16 <= T_total:
            chunk = frames[:, i : i+16]
            keep_len = 16
            is_last = False
        else:
            # For the last chunk, take the last 16 frames of the video to avoid padding
            chunk = frames[:, -16:]
            keep_len = T_total % 16
            # If the video is shorter than 16 frames, handle appropriately
            if T_total < 16:
                # We can pad by repeating the last frame to reach 16
                padding_needed = 16 - T_total
                last_frame = frames[:, -1:]
                chunk = torch.cat([frames, last_frame.repeat(1, padding_needed, 1, 1)], dim=1)
                keep_len = T_total
            is_last = True
            
        # Run inference
        # Input shape: [1, 3, 16, 224, 224]
        with torch.no_grad():
            _, _, _, vis = model(chunk.unsqueeze(0).to(device), 1, mask_ratio=mask_ratio, visualize=True)
            
        # vis shape: [1, 3, 3, 16, 224, 224]
        # vis[0] has shape [3, 3, 16, 224, 224]
        orig = vis[0][0].cpu()       # [3, 16, 224, 224]
        mask_c = vis[0][1].cpu()     # [3, 16, 224, 224]
        recon = vis[0][2].cpu()      # [3, 16, 224, 224]
        
        if is_last:
            if T_total < 16:
                orig = orig[:, :keep_len]
                mask_c = mask_c[:, :keep_len]
                recon = recon[:, :keep_len]
            else:
                orig = orig[:, -keep_len:]
                mask_c = mask_c[:, -keep_len:]
                recon = recon[:, -keep_len:]
                
        all_originals.append(orig)
        all_masked.append(mask_c)
        all_reconstructs.append(recon)
        
    # Concatenate all chunk outputs along the temporal dimension
    all_originals = torch.cat(all_originals, dim=1)
    all_masked = torch.cat(all_masked, dim=1)
    all_reconstructs = torch.cat(all_reconstructs, dim=1)
    
    return all_originals, all_masked, all_reconstructs

def unnormalize(tensor):
    """
    Un-normalizes a video tensor from MEAN and STD back to standard RGB [0, 255] uint8.
    Input shape: [3, T, 224, 224]
    Output shape: [T, 224, 224, 3]
    """
    tensor = tensor.permute(1, 2, 3, 0) # [T, 224, 224, 3]
    tensor = tensor * torch.tensor(STD) + torch.tensor(MEAN)
    tensor = torch.clip(tensor * 255, 0, 255).to(torch.uint8)
    return tensor

def main():
    args = parse_args()
    
    print("Loading video...")
    raw_frames, fps = load_video_frames(args.video_path)
    
    # Pre-process frames: normalize, permute, and spatial sampling
    print("Pre-processing video frames...")
    frames = tensor_normalize(raw_frames, torch.tensor(MEAN), torch.tensor(STD)).permute(3, 0, 1, 2)
    frames = spatial_sampling(
        frames,
        spatial_idx=1,
        min_scale=256,
        max_scale=256,
        crop_size=224,
    )
    
    # Load model and checkpoint
    print(f"Initializing model and loading checkpoint from {args.checkpoint}...")
    model = mae_vit_large_patch16(decoder_embed_dim=512, decoder_depth=4, mask_type="st", t_patch_size=2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print("Model load status:", msg)
    
    # Run inference for the entire video
    print(f"Running inference on {args.device} with mask ratio {args.mask_ratio}...")
    orig_t, masked_t, recon_t = process_video_mae(frames, model, args.mask_ratio, args.device)
    
    # Un-normalize back to [0, 255] uint8
    print("Un-normalizing resulting frames...")
    orig_frames = unnormalize(orig_t)
    masked_frames = unnormalize(masked_t)
    recon_frames = unnormalize(recon_t)
    
    # Concatenate horizontally: Original | Masked | Reconstructed
    print("Generating side-by-side visualization...")
    combined_frames = torch.cat([orig_frames, masked_frames, recon_frames], dim=2) # Concatenate along width
    
    # Write to output video file using PyAV for maximum compatibility and robustness
    print(f"Saving side-by-side video to {args.output_path}...")
    container = av.open(args.output_path, mode='w')
    
    # Try using 'h264' first, fallback to 'mpeg4'
    try:
        stream = container.add_stream('h264', rate=int(fps))
    except Exception:
        stream = container.add_stream('mpeg4', rate=int(fps))
        
    height, width = combined_frames.shape[1], combined_frames.shape[2]
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    
    for frame in combined_frames:
        frame_np = frame.numpy()
        av_frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for packet in stream.encode(av_frame):
            container.mux(packet)
            
    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)
        
    container.close()
    print("Finished successfully!")

if __name__ == "__main__":
    main()