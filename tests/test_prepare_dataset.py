import os, pathlib, subprocess, sys, tempfile, unittest

PREPARE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "prepare_dataset.py")

class PrefixTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for n in ("train10_s_000", "train10_s_001", "train100_s_000", "val_s_000"): pathlib.Path(self.dir, f"{n}.mp4").touch()

    def run_prepare(self, *args): return subprocess.run([sys.executable, PREPARE, self.dir, *args], capture_output=True, text=True)

    def test_each_prefix_gets_its_own_csv(self):
        self.assertEqual(self.run_prepare("--prefix", "train10", "val").returncode, 0)
        self.assertEqual(sorted(f for f in os.listdir(self.dir) if f.endswith(".csv")), ["train10.csv", "val.csv"])
        rows = pathlib.Path(self.dir, "train10.csv").read_text().splitlines()
        self.assertEqual([os.path.basename(r.split()[0]) for r in rows], ["train10_s_000.mp4", "train10_s_001.mp4"])

    def test_prefix_and_split_exclude_each_other(self):
        self.assertEqual(self.run_prepare("--split", "prefix", "--prefix", "val").returncode, 2)

if __name__ == "__main__": unittest.main()
