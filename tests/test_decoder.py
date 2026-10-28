import tempfile, unittest
import av, numpy as np
from tests import _package  # noqa: F401
from mae_st.util.decoder import decoder

FRAMES, H, W, FPS, STEP = 30, 48, 80, 10, 8

def make_video(first_pts=0):
    path = tempfile.mktemp(suffix=".mp4")
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=FPS); s.width, s.height, s.pix_fmt = W, H, "yuv444p"; s.options = {"qp": "0"}
        for i in range(FRAMES):
            f = av.VideoFrame.from_ndarray(np.full((H, W, 3), i * STEP, np.uint8), format="rgb24"); f.pts = first_pts + i
            c.mux(s.encode(f))
        c.mux(s.encode())
    return open(path, "rb").read()

def frame_ids(frames): return [round(float(v) / STEP) for v in frames.float().mean(dim=(1, 2, 3))]

class DecoderTest(unittest.TestCase):
    video = make_video()

    def decode(self, **kw):
        return decoder.decode(self.video, sampling_rate=1, num_frames=10, num_clips=1, video_meta={}, target_fps=FPS, **kw)

    def test_full_decode_returns_every_frame_in_order(self):
        frames, fps, all_video = self.decode(clip_idx=-1)
        self.assertEqual((tuple(frames.shape), fps, all_video), ((FRAMES, H, W, 3), FPS, True))
        self.assertEqual(frame_ids(frames), list(range(FRAMES)))

    def test_selective_decode_returns_a_contiguous_clip(self):
        frames, _, all_video = self.decode(clip_idx=0, rigid_decode_all_video=False)
        ids = frame_ids(frames)
        self.assertFalse(all_video)
        self.assertGreaterEqual(len(ids), 10)
        self.assertEqual(ids, list(range(ids[0], ids[0] + len(ids))))

    def test_selective_decode_counts_from_the_first_pts(self):
        # streams often start at a non-zero pts (encoder delay, edit lists); the clip must still start at frame 0
        frames, _, _ = decoder.decode(make_video(first_pts=2), sampling_rate=1, num_frames=10, clip_idx=0, num_clips=3, video_meta={}, target_fps=FPS, rigid_decode_all_video=False)
        self.assertEqual(frame_ids(frames)[:10], list(range(10)))

    def test_min_dimension_resizes_the_shorter_side(self):
        frames, _, _ = self.decode(clip_idx=-1, max_spatial_scale=24)
        self.assertEqual(tuple(frames.shape[1:3]), (24, 40))

    def test_undecodable_input_returns_none_frames(self):
        frames, _, _ = decoder.decode(b"not a video", 1, 10, video_meta={})
        self.assertIsNone(frames)

if __name__ == "__main__": unittest.main()
