import os, tempfile
import av, numpy as np

def write_video(path, frames, fps=10):
    t, h, w = frames.shape[:3]
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps); s.width, s.height, s.pix_fmt = w, h, "yuv444p"; s.options = {"qp": "0"}
        for f in frames: c.mux(s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")))
        c.mux(s.encode())

def write_dataset(videos, splits=("train", "val", "test")):
    root = tempfile.mkdtemp()
    paths = []
    for i, frames in enumerate(videos):
        paths.append(os.path.join(root, f"video_{i:03d}.mp4")); write_video(paths[-1], frames)
    for split in splits:
        with open(os.path.join(root, f"{split}.csv"), "w") as f: f.writelines(f"{p} 0\n" for p in paths)
    return root

def decoded(path):
    with av.open(path) as c: return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
