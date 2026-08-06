# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import copy
import sys
import tempfile
from io import BytesIO

import av

def get_video_container(handle, multi_thread_decode=False):
    # Use local data.
    with open(handle, "rb") as fp:
        container = fp.read()
    return container
