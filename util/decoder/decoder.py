# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import io
import math
import random
from fractions import Fraction

import av
import numpy as np
import torch


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    return new_frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    max_spatial_scale=0,
    use_offset=False,
    rigid_decode_all_video=True,
    modalities=("visual",),
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. 0 keeps the native
            resolution.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    fps, decode_all_video = None, True
    try:
        assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
        video_start_pts, video_end_pts = 0, -1
        # The video_meta is empty, fetch the meta data from the raw video.
        if len(video_meta) == 0:
            video_meta.update(_probe_video(container))

        fps = video_meta["video_fps"]
        if not rigid_decode_all_video:
            if (
                video_meta["has_video"]
                and video_meta["video_denominator"] > 0
                and video_meta["video_duration"] > 0
            ):
                # try selective decoding.
                decode_all_video = False
                clip_size = sampling_rate * num_frames / target_fps * fps
                start_idx, end_idx = get_start_end_idx(
                    fps * video_meta["video_duration"],
                    clip_size,
                    clip_idx,
                    num_clips,
                    use_offset=use_offset,
                )
                # Convert frame index to pts, counted from the stream's first pts.
                pts_per_frame = video_meta["video_denominator"] / fps
                video_start_pts = video_meta["video_start_pts"] + int(start_idx * pts_per_frame)
                video_end_pts = video_meta["video_start_pts"] + int(end_idx * pts_per_frame)

        v_frames = _read_video(container, video_start_pts, video_end_pts, max_spatial_scale)
        if v_frames.shape == torch.Size([0]):
            # failed selective decoding
            decode_all_video = True
            v_frames = _read_video(container, 0, -1, max_spatial_scale)
    except Exception as e:
        print("Failed to decode with exception: {}".format(e))
        return None, fps, decode_all_video

    # Return None if the frames was not decoded successfully.
    if v_frames is None or v_frames.size(0) == 0:
        return None, fps, decode_all_video
    return v_frames, fps, decode_all_video


# PyAV replacements for torchvision.io._probe_video_from_memory and
# _read_video_from_memory, the private in-memory video_reader API that the
# upstream decoder used. Stock torchvision wheels do not ship that backend, and
# torchvision 0.26 removed the API.
def _probe_video(video_bytes):
    with av.open(io.BytesIO(video_bytes)) as c:
        s = c.streams.video[0]
        tb = s.time_base or Fraction(0, 1)
        duration = float(s.duration * tb) if s.duration else (c.duration or 0) / av.time_base
        return dict(has_video=True, video_timebase=tb, video_numerator=tb.numerator, video_denominator=tb.denominator, video_start_pts=s.start_time or 0, video_duration=duration, video_fps=float(s.average_rate or 0))


def _read_video(video_bytes, start_pts=0, end_pts=-1, min_dimension=0):
    frames = []
    with av.open(io.BytesIO(video_bytes)) as c:
        s = c.streams.video[0]
        if start_pts > 0: c.seek(start_pts, stream=s, backward=True, any_frame=False)
        for f in c.decode(s):
            if f.pts is not None and f.pts < start_pts: continue
            if end_pts != -1 and f.pts is not None and f.pts > end_pts: break
            frames.append(_rgb(f, min_dimension))
    return torch.from_numpy(np.stack(frames)) if frames else torch.empty(0)


# min_dimension > 0 resizes the shorter side to it, keeping the aspect ratio, as
# torchvision's video_min_dimension did (libswscale bilinear here, so pixel values
# may differ slightly from torchvision's build).
def _rgb(frame, min_dimension):
    if min_dimension <= 0: return frame.to_ndarray(format="rgb24")
    k = min_dimension / min(frame.height, frame.width)
    return frame.reformat(width=round(frame.width * k), height=round(frame.height * k), format="rgb24", interpolation="BILINEAR").to_ndarray()
