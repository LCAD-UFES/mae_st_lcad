"""Prepare a small slice of Kinetics-400 for mae_st, per upstream DATASET.md:
resize videos to short edge 256 and emit train/val/test.csv as "path label".

    python tools/prepare_k400_slice.py [--out-dir DIR] [--raw-dir DIR] [--n-videos N]
                                       [--partial-mb MB]

Source: the CVD Foundation S3 mirror of Kinetics-400 (annotations + the
validation tarball part_0). Only the first --partial-mb megabytes of the
tarball are downloaded: tar+gzip is a stream, so that prefix already holds
dozens of complete videos; the truncated tail is detected by decoding.

Label ids: the annotation file carries label *names*; mae_st's Kinetics
loader wants an int. Standard convention: sort the 400 unique names
alphabetically and number them 0..399. The mapping is written to labels.txt.

Split choice: all N videos go into each of train/val/test.csv -- this slice
exists for smoke and small training tests, not for measuring generalization.
Edit the csvs by hand for a real split.
"""

import argparse
import csv
import os
import os.path as osp
import subprocess
from fractions import Fraction

import av

S3 = "https://s3.amazonaws.com/kinetics/400"
SHORT_EDGE = 256


def download_raw(raw_dir, partial_mb):
    """Annotations + a prefix of val/part_0.tar.gz, extracted into raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    ann = osp.join(raw_dir, "val_annotations.csv")
    if not osp.isfile(ann):
        subprocess.run(["curl", "-s", "-o", ann, f"{S3}/annotations/val.csv"], check=True)
    if not any(f.endswith(".mp4") for f in os.listdir(raw_dir)):
        n_bytes = partial_mb * 1024 * 1024
        # tar exits non-zero on the truncated stream; the complete members are kept
        subprocess.run(
            f"curl -s -r 0-{n_bytes} {S3}/val/part_0.tar.gz | tar -xzf - -C '{raw_dir}' 2>/dev/null",
            shell=True)
    return ann


def resize_video(src, dst):
    """decode -> scale short edge to 256 -> h264 (PyAV's bundled libx264,
    no ffmpeg CLI needed)."""
    with av.open(src) as in_c:
        in_s = in_c.streams.video[0]
        w, h = in_s.codec_context.width, in_s.codec_context.height
        scale = SHORT_EDGE / min(w, h)
        nw, nh = (round(w * scale / 2) * 2, round(h * scale / 2) * 2)  # yuv420p needs even dims
        fps = in_s.average_rate or Fraction(30, 1)
        with av.open(dst, "w") as out_c:
            out_s = out_c.add_stream("libx264", rate=fps)
            out_s.width, out_s.height = nw, nh
            out_s.pix_fmt = "yuv420p"
            out_s.options = {"crf": "18", "preset": "veryfast"}
            n = 0
            for frame in in_c.decode(video=0):
                frame = frame.reformat(width=nw, height=nh, format="yuv420p")
                frame.pts = n  # re-time explicitly in the codec time_base of 1/fps
                frame.time_base = Fraction(1, 1) / fps
                out_c.mux(out_s.encode(frame))
                n += 1
            out_c.mux(out_s.encode())  # flush
    return nw, nh, n


def fully_decodable(path):
    """Reject truncated files (the tarball download was partial)."""
    try:
        with av.open(path) as c:
            return sum(1 for _ in c.decode(video=0)) > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="~/datasets/kinetics400_slice10")
    parser.add_argument("--raw-dir", default="~/datasets/kinetics400_raw",
                        help="where the annotations and raw videos are downloaded")
    parser.add_argument("--n-videos", type=int, default=10)
    parser.add_argument("--partial-mb", type=int, default=150,
                        help="how much of the tarball to download")
    args = parser.parse_args()
    out_dir = osp.expanduser(args.out_dir)
    raw_dir = osp.expanduser(args.raw_dir)
    videos_dir = osp.join(out_dir, "videos")

    annotations = download_raw(raw_dir, args.partial_mb)

    # --- label name -> int id (alphabetical over the full 400-class set) -----
    with open(annotations) as f:
        rows = list(csv.DictReader(f))
    label_names = sorted({r["label"] for r in rows})
    assert len(label_names) == 400, f"expected 400 classes, got {len(label_names)}"
    label_id = {name: i for i, name in enumerate(label_names)}
    stem_label = {  # "{youtube_id}_{start:06d}_{end:06d}" -> label name
        f'{r["youtube_id"]}_{int(r["time_start"]):06d}_{int(r["time_end"]):06d}': r["label"]
        for r in rows
    }

    # --- pick N valid videos, preferring distinct classes --------------------
    candidates = sorted(f for f in os.listdir(raw_dir) if f.endswith(".mp4"))
    picked, used_labels = [], set()
    for fname in candidates:
        if len(picked) == args.n_videos:
            break
        label = stem_label.get(fname[:-4])
        if label is None or label in used_labels:
            continue
        if not fully_decodable(osp.join(raw_dir, fname)):
            print(f"skip (truncated/undecodable): {fname}")
            continue
        picked.append((fname, label))
        used_labels.add(label)
    assert len(picked) == args.n_videos, (
        f"only {len(picked)} usable videos found; raise --partial-mb")

    # --- resize into the dataset dir and write csvs --------------------------
    os.makedirs(videos_dir, exist_ok=True)
    entries = []
    for fname, label in picked:
        dst = osp.join(videos_dir, fname)
        nw, nh, n = resize_video(osp.join(raw_dir, fname), dst)
        entries.append((dst, label_id[label]))
        print(f"{fname}: {label!r} (id {label_id[label]}) -> {nw}x{nh}, {n} frames")

    with open(osp.join(out_dir, "labels.txt"), "w") as f:
        for name in label_names:
            f.write(f"{label_id[name]} {name}\n")
    for split in ("train", "val", "test"):
        with open(osp.join(out_dir, f"{split}.csv"), "w") as f:
            for path, lid in entries:
                f.write(f"{path} {lid}\n")
    print(f"\nwrote {len(entries)} videos + train/val/test.csv + labels.txt in {out_dir}")


if __name__ == "__main__":
    main()
