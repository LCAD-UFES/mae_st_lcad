"""Stack-compatibility shims for running the pristine ~/mae_st clone.

The upstream repo assumes Meta's internal software stack; on ours
(torch 2.11, torchvision 0.26, no scikit-learn) it breaks in four places.
Each shim here patches one of them from the OUTSIDE, keeping the clone
untouched. tools/run_test_smoke.py carries the same shims inline with a
longer didactic commentary; this module is the reusable form every driver
in this workspace imports.

Call compat.apply() once, BEFORE importing anything from mae_st.
"""

import argparse
import importlib.machinery
import io as _io
import sys
import types
from fractions import Fraction

import av
import numpy as np
import torch
from torchvision import io as tv_io

_applied = False


def apply():
    """Install all shims (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True
    _stub_meta_internal_modules()
    _allowlist_checkpoint_globals()
    _patch_torchvision_video_io()


# -- shim 1+2: unsatisfiable module-level imports ---------------------------
# util/misc.py:23 imports torch.fb.rendezvous.zeus (Meta-internal, used only
# for its import side effect) and util/meters.py:7 imports sklearn (absent
# from the curated env; only reached via the multi-label get_map() path).
# Pre-registering stubs in sys.modules satisfies both imports. The __spec__
# is required because torch._dynamo.trace_rules calls find_spec on every
# entry of sys.modules and raises on modules without one.


def _stub_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = mod
    return mod


def _sklearn_stub(*_a, **_k):
    raise RuntimeError("sklearn stubbed out by custom/compat.py; "
                       "multi-label metrics are not supported")


def _stub_meta_internal_modules():
    for name in ("torch.fb", "torch.fb.rendezvous", "torch.fb.rendezvous.zeus"):
        if name not in sys.modules:
            _stub_module(name)
    if "sklearn" not in sys.modules:
        sk = _stub_module("sklearn")
        sk_metrics = _stub_module("sklearn.metrics")
        sk_metrics.average_precision_score = _sklearn_stub
        sk.metrics = sk_metrics


# -- shim 3: torch.load safety gate (torch >= 2.6) --------------------------
# Checkpoints in this project store their argparse.Namespace under "args";
# weights_only=True (the default since torch 2.6) rejects that type unless
# allowlisted. We trust our own checkpoints.


def _allowlist_checkpoint_globals():
    torch.serialization.add_safe_globals([argparse.Namespace])


# -- shim 4: the video decoder ----------------------------------------------
# util/decoder/decoder.py decodes through torchvision's PRIVATE in-memory
# API (_probe_video_from_memory / _read_video_from_memory), removed in
# torchvision 0.26. Reimplemented on PyAV. decoder.py holds a reference to
# the torchvision.io module object (`import torchvision.io as io`), so
# assigning attributes on it is enough; DataLoader workers fork and inherit
# the patch.


def _open_video(video_tensor):
    return av.open(_io.BytesIO(video_tensor.numpy().tobytes()))


def _probe_video_from_memory(video_tensor):
    with _open_video(video_tensor) as container:
        stream = container.streams.video[0]
        timebase = stream.time_base or Fraction(0, 1)
        duration = float((stream.duration or 0) * timebase)  # seconds
        fps = float(stream.average_rate) if stream.average_rate else 0.0
    return types.SimpleNamespace(
        video_timebase=timebase,
        has_video=True,
        video_duration=duration,
        video_fps=fps,
        audio_timebase=Fraction(0, 1),
        has_audio=False,
        audio_duration=0.0,
        audio_sample_rate=0.0,
    )


def _read_video_from_memory(video_tensor, video_pts_range=(0, -1), **_kwargs):
    start_pts, end_pts = video_pts_range
    frames = []
    with _open_video(video_tensor) as container:
        for i, frame in enumerate(container.decode(video=0)):
            pts = frame.pts if frame.pts is not None else i
            if pts < start_pts:
                continue
            if end_pts != -1 and pts > end_pts:
                break
            frames.append(frame.to_rgb().to_ndarray())  # H W C uint8
    if not frames:
        return torch.empty(0), None
    return torch.from_numpy(np.stack(frames)), None  # T H W C uint8


def _patch_torchvision_video_io():
    tv_io._probe_video_from_memory = _probe_video_from_memory
    tv_io._read_video_from_memory = _read_video_from_memory
