"""Write the train/val/test csvs mae_st's Kinetics loader reads, for a folder of videos.

    python tools/prepare_dataset.py <videos_dir> [--split all|prefix] [--limit N] [--min-frames K] [--out DIR]

The loader reads <dataset>/{train,val,test}.csv, one `absolute_path label` per
line. Absolute paths tie the csvs to the machine, so run this where training
runs, after copying the videos. Typical input is the output folder of
lidar_sweep_viewer2 (lsv2), which writes clips and no csvs.

--split all     every video in all three csvs: the overfit protocol, where
                val loss is measured on the training clips (default)
--split prefix  by file-name prefix, `train_`, `val_`, `test_` (lsv2's `prefix`
                job key); a split with no videos is an error
--limit N       keep the first N videos of each split, in file-name order
--min-frames K  drop videos with fewer than K frames, e.g. a trailing partial
                clip shorter than the config's num_frames
"""

import argparse
import glob
import os
import os.path as osp
import sys

import av

SPLITS = ("train", "val", "test")
LABEL = 0  # pretraining ignores labels, but the loader parses one per line


def frame_count(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return s.frames or sum(1 for _ in c.decode(s))


def select(videos, split, mode):
    if mode == "all": return videos
    return [v for v in videos if osp.basename(v).startswith(split + "_")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos_dir")
    p.add_argument("--split", choices=("all", "prefix"), default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-frames", type=int, default=0)
    p.add_argument("--out", default=None, help="where the csvs go (default: videos_dir)")
    a = p.parse_args(argv)
    videos = sorted(osp.abspath(v) for v in glob.glob(osp.join(a.videos_dir, "*.mp4")))
    if not videos: sys.exit(f"no .mp4 files in {a.videos_dir}")
    if any(" " in v for v in videos): sys.exit("video paths must not contain spaces: the csv format separates path and label with one")
    short = [v for v in videos if a.min_frames and frame_count(v) < a.min_frames]
    for v in short: print(f"dropped (fewer than {a.min_frames} frames): {osp.basename(v)}")
    videos = [v for v in videos if v not in short]
    out = a.out or a.videos_dir
    os.makedirs(out, exist_ok=True)
    for split in SPLITS:
        chosen = select(videos, split, a.split)[: a.limit]
        if not chosen: sys.exit(f"no videos for split {split!r} (--split {a.split})")
        with open(osp.join(out, f"{split}.csv"), "w") as f: f.writelines(f"{v} {LABEL}\n" for v in chosen)
        print(f"{split}.csv: {len(chosen)} videos")


if __name__ == "__main__":
    main()
