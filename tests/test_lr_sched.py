import argparse, math, unittest
from tests import _package  # noqa: F401
from mae_st.util import lr_sched

def lr_at(epoch, **kw):
    args = argparse.Namespace(**{**dict(lr=1e-3, min_lr=1e-4, warmup_epochs=5, epochs=3000), **kw})
    groups = [{}, {"lr_scale": 0.5}]
    lr = lr_sched.adjust_learning_rate(argparse.Namespace(param_groups=groups), epoch, args)
    assert groups == [{"lr": lr}, {"lr_scale": 0.5, "lr": lr * 0.5}]
    return lr

def upstream(e, lr=1e-3, min_lr=1e-4, w=5, E=3000):
    # the formula before lr_cycles, operation for operation
    if e < w: return lr * e / w
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (e - w) / (E - w)))

class LrCyclesTest(unittest.TestCase):
    def test_one_cycle_is_upstream_bit_for_bit(self):
        for e in [i / 18 for i in range(3000 * 18)]:
            self.assertEqual(lr_at(e), upstream(e))
            self.assertEqual(lr_at(e, lr_cycles=1), upstream(e))

    def test_each_cycle_decays_to_min_lr_and_restarts(self):
        for n in (2, 5, 6):
            period = 2995 / n
            for k in range(n):
                start = 5 + k * period
                self.assertAlmostEqual(lr_at(start + 1e-9, lr_cycles=n), 1e-3)
                self.assertAlmostEqual(lr_at(start + period / 2, lr_cycles=n), 5.5e-4)
                self.assertAlmostEqual(lr_at(start + period - 1e-9, lr_cycles=n), 1e-4)
            per_epoch = [lr_at(e, lr_cycles=n) for e in range(5, 3000)]
            self.assertEqual(sum(b > a for a, b in zip(per_epoch, per_epoch[1:])), n - 1)

    def test_warmup_is_not_repeated(self):
        self.assertEqual([lr_at(e, lr_cycles=4) for e in range(6)], [upstream(e) for e in range(6)])

if __name__ == "__main__": unittest.main()
