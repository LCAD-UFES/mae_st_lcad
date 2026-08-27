"""Smoke-test launcher for the pristine facebookresearch/mae_st clone.

Goal: execute the repo's own run_test.py end-to-end WITHOUT editing the clone.
The upstream code was published straight out of Meta's internal monorepo and
assumes an internal software stack; on our stack (torch 2.11, torchvision 0.26,
no scikit-learn) it breaks in four places. Each break is patched here, from the
outside, in the order Python would hit it:

  section 1  util/misc.py:23        imports torch.fb.rendezvous.zeus (Meta-internal)
  section 2  util/meters.py:7       imports sklearn (not in the curated env)
  section 3  main_test.py:228       torch.load rejects the checkpoint (torch >= 2.6)
  section 4  util/decoder/decoder.py uses torchvision private video APIs
                                     that were removed in torchvision 0.26
  section 5  builds argv and runs run_test.py as if typed on the command line

Everything here is a *workaround*, deliberately kept outside the repository.
The durable fix for each item belongs in the repository itself and is noted in
the corresponding section.

Usage (from any directory, with the environment active):
  PYTHONPATH=$HOME python tools/run_test_smoke.py

PYTHONPATH must hold the repo's PARENT directory so that main_test.py's
`import mae_st.util...` package imports resolve; the repo root itself (needed
by run_test.py's flat `from main_test import ...`) is inserted into sys.path
by section 5 below.
"""

import argparse
import importlib.machinery
import io as _io
import os.path as osp
import runpy
import sys
import types
from fractions import Fraction

import av
import numpy as np
import torch
from torchvision import io as tv_io

# ---------------------------------------------------------------------------
# Helper: build a fake ("stub") module and register it in sys.modules.
#
# Python's import system checks sys.modules first: if `import x.y` finds "x.y"
# already registered, it uses that object and never looks for a real package.
# Pre-registering stubs therefore makes an unsatisfiable import succeed.
#
# The __spec__ matters: torch._dynamo.trace_rules walks sys.modules calling
# importlib.util.find_spec on each name, which raises ValueError on any module
# whose __spec__ is None. A bare types.ModuleType has none, so we attach one.
# ---------------------------------------------------------------------------


def _stub_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Section 1 -- util/misc.py:23: `import torch.fb.rendezvous.zeus`
#
# torch.fb is Meta's internal extension of PyTorch; the zeus module registers
# their cluster rendezvous backend as an import side effect. The symbol is
# never referenced again anywhere in the repo, so an empty stub is enough.
# `torch` itself must be imported first (top of file) so the real package is
# in sys.modules before we hang fake submodules off its name.
#
# Durable fix: delete the import line. (The LCAD fork did exactly this.)
# ---------------------------------------------------------------------------

for _name in ("torch.fb", "torch.fb.rendezvous", "torch.fb.rendezvous.zeus"):
    _stub_module(_name)

# ---------------------------------------------------------------------------
# Section 2 -- util/meters.py:7: `from sklearn.metrics import average_precision_score`
#
# scikit-learn is not in the curated conda env. The imported function is only
# reached through get_map(), the multi-label mAP path; our test is single-label
# (main_test.py passes multilabel=False to TestMeter), so that path never runs.
# The stub still provides the symbol -- raising loudly if ever called, instead
# of failing silently ("fail loudly" working agreement).
#
# Durable fix: either guard the import or add scikit-learn to the manifest.
# ---------------------------------------------------------------------------


def _sklearn_stub(*_a, **_k):
    raise RuntimeError(
        "sklearn stubbed out by run_test_smoke.py; multi-label path not supported"
    )


_sk = _stub_module("sklearn")
_sk_metrics = _stub_module("sklearn.metrics")
_sk_metrics.average_precision_score = _sklearn_stub
_sk.metrics = _sk_metrics

# ---------------------------------------------------------------------------
# Section 3 -- torch.load safety gate (torch >= 2.6)
#
# The checkpoint stores, besides the weights, the argparse.Namespace it was
# trained with (key "args"). Since torch 2.6, torch.load defaults to
# weights_only=True, which only unpickles an allowlist of known-safe types;
# Namespace is not on it, so main_test.py:228 raises UnpicklingError.
# We trust this file (downloaded from Meta, already used by the demo), so we
# extend the allowlist with the official API instead of patching torch.load.
#
# Durable fix: pass weights_only=False (or this allowlist) in util/misc.py's
# checkpoint-loading helpers.
# ---------------------------------------------------------------------------

torch.serialization.add_safe_globals([argparse.Namespace])

# ---------------------------------------------------------------------------
# Section 4 -- the video decoder (the substantive one)
#
# util/decoder/decoder.py decodes video through torchvision's PRIVATE in-memory
# API: io._probe_video_from_memory (read metadata) and io._read_video_from_memory
# (decode frames). Both were removed in modern torchvision; on 0.26 the call
# raises AttributeError, which decoder.decode() swallows (`return None`), and
# kinetics.py:267 then crashes far from the cause with
# "TypeError: cannot unpack non-iterable NoneType object".
#
# We reimplement both functions on PyAV (`import av`), which decodes the video
# fine, and monkeypatch them onto torchvision.io. decoder.py did
# `import torchvision.io as io`, i.e. it holds a reference to the same module
# object we patch -- so it transparently picks up our versions. DataLoader
# workers are forked from this process on Linux, so they inherit the patch.
#
# What the two functions must honor, mirroring the old torchvision contract:
#  - _probe returns an object with the metadata fields decoder.decode() reads
#    (video timebase / fps / duration, plus audio fields it stores but ignores).
#  - _read receives the raw file bytes as a uint8 tensor plus a pts range
#    (start, end) in stream-timebase units; (0, -1) means "decode everything".
#    It returns (frames as a [T, H, W, C] uint8 tensor, audio) -- audio unused.
#    The pts range is how decode() implements SELECTIVE decoding: for test
#    clip k of 10, it only wants that temporal slice of the video.
#
# Durable fix: port the decoder to a public backend (PyAV, as PySlowFast does).
# ---------------------------------------------------------------------------


def _open_video(video_tensor):
    """The pipeline reads the file as raw bytes (see video_container.py);
    wrap those bytes in a file-like object and hand them to PyAV."""
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


tv_io._probe_video_from_memory = _probe_video_from_memory
tv_io._read_video_from_memory = _read_video_from_memory

# ---------------------------------------------------------------------------
# Section 5 -- run the unmodified run_test.py
#
# runpy.run_path executes a script as __main__, but unlike `python script.py`
# it does NOT put the script's directory on sys.path -- so we add the repo root
# ourselves (run_test.py does `from main_test import ...`, a flat import).
# The mae_st.* package imports inside main_test.py are satisfied by PYTHONPATH
# containing the repo's PARENT directory (see Usage in the docstring).
#
# The argument choices that matter:
#   --num_frames 16 --t_patch_size 2   MUST match the checkpoint: they shape
#                                      patch_embed (Conv3d kernel) and pos_embed
#                                      (1, 8*14*14=1568, 1024). The parser
#                                      defaults (32 / 4) would be a hard
#                                      RuntimeError shape mismatch on load.
#   no --sep_pos_embed, no --cls_embed the checkpoint has a single joint
#                                      pos_embed and no class token; both flags
#                                      default False in main_test's parser.
#   --finetune <pretrain checkpoint>   a PRETRAIN checkpoint, so this is a
#                                      machinery test only: the classifier head
#                                      is random and top-1 is meaningless.
#   --batch_size 3                     30 test views (10 temporal clips x 3
#                                      spatial crops of the single video) in
#                                      10 iterations.
# ---------------------------------------------------------------------------

# Same locations bootstrapper.py uses, spelled out here so this file stays
# self-contained (importing bootstrapper would silently pre-apply the very
# shims this script exists to show).
import os

REPO_ROOT = osp.abspath(osp.expanduser(os.environ.get("MAE_ST_UPSTREAM", "~/mae_st")))
EXPERIMENTS_ROOT = osp.abspath(osp.expanduser(os.environ.get("MAE_ST_EXPERIMENTS", "~/mae_st_experiments")))
DATASETS_ROOT = osp.abspath(osp.expanduser(os.environ.get("MAE_ST_DATASETS", "~/datasets")))
sys.path.insert(0, REPO_ROOT)

sys.argv = [
    "run_test.py",
    "--path_to_data_dir", osp.join(DATASETS_ROOT, "kinetics400_slice10"),
    "--finetune", osp.join(EXPERIMENTS_ROOT, "checkpoints/video-mae-200x4-nonorm.pth"),
    "--model", "vit_large_patch16",
    "--nb_classes", "400",
    "--num_frames", "16",
    "--t_patch_size", "2",
    "--sampling_rate", "4",
    "--batch_size", "3",
    "--num_workers", "4",
    "--output_dir", osp.join(EXPERIMENTS_ROOT, "work_dirs/mae_st_overfit_tests/smoke_run_test_original"),
]

runpy.run_path(f"{REPO_ROOT}/run_test.py", run_name="__main__")
