import os
import sys
import unittest
import numpy as np
import pandas as pd

GT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.insert(0, GT_DIR)

from attack_injector import inject
from train import split_captures, is_test_graph, partition_benign


def _stream(seconds=14.0):
    t = np.arange(0, seconds, 0.01)
    return pd.DataFrame({"Label": 0, "Time": t, "ID": 208, "Signal_1_of_ID": np.sin(t)})


class TestInjector(unittest.TestCase):
    def test_fabrication_adds_labelled_frames_and_keeps_legit(self):
        df = _stream()
        out = inject(df, "fabrication", 2, 208, np.random.default_rng(0))
        self.assertEqual(int((out.Label == 0).sum()), len(df))
        self.assertEqual(int((out.Label == 1).sum()), 2 * 1000)  # 1000 legit frames in the 10 s episode
        self.assertTrue((out.loc[out.Label == 1, "Signal_1_of_ID"] == df["Signal_1_of_ID"].max()).all())

    def test_ids_stay_integer_for_every_attack(self):
        for attack, level in (("fabrication", 1), ("drift", 0.1), ("flooding", 10), ("suspension", 0.5)):
            out = inject(_stream(), attack, level, 208, np.random.default_rng(0))
            self.assertEqual(str(out["ID"].dtype), "int64", attack)
            self.assertNotIn(".", out.to_csv(index=False).splitlines()[1].split(",")[2], attack)

    def test_suspension_removes_frames_only_inside_episode(self):
        df = _stream()
        out = inject(df, "suspension", 1.0, 208, np.random.default_rng(0))
        self.assertEqual(int(((out.Time >= 2) & (out.Time < 12)).sum()), 0)
        self.assertEqual(len(out), len(df) - 1000)

    def test_drift_slows_cadence(self):
        out = inject(_stream(), "drift", 0.10, 208, np.random.default_rng(0))
        ep = out[(out.Time >= 2) & (out.Time < 12)]
        self.assertTrue((ep.Label == 1).all())
        self.assertGreater(np.median(np.diff(ep.Time.values)), 0.0105)

    def test_flooding_uses_flood_id(self):
        out = inject(_stream(), "flooding", 100, 208, np.random.default_rng(0))
        self.assertEqual(int((out.ID == 0).sum()), 1000)


class TestFoldSplit(unittest.TestCase):
    def test_every_capture_tested_once_and_splits_disjoint(self):
        caps = [f"ambient_{i}" for i in range(12)]
        tested = []
        for f in range(6):
            tr, va, te = split_captures(caps, fold=f, n_folds=6)
            self.assertFalse(set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            self.assertEqual(len(tr) + len(va) + len(te), 12)
            tested += te
        self.assertEqual(sorted(tested), sorted(caps))

    def test_synthetic_only_in_parent_test_fold(self):
        class G:
            def __init__(self, name, y):
                self.capture_name, self.y = name, type("T", (), {"item": lambda s: y})()
        self.assertTrue(is_test_graph(G("synthetic__ambient_1__drift_0.1", 1), ["ambient_1"]))
        self.assertFalse(is_test_graph(G("synthetic__ambient_1__drift_0.1", 0), ["ambient_1"]))
        self.assertFalse(is_test_graph(G("synthetic__ambient_2__drift_0.1", 1), ["ambient_1"]))
        self.assertTrue(is_test_graph(G("max_speedometer_attack_1_masquerade", 0), ["ambient_1"]))
        self.assertFalse(is_test_graph(G("ambient_2", 0), ["ambient_1"]))

    def test_calibration_tail_disjoint_and_no_shared_messages(self):
        class G:
            def __init__(self, name, start):
                self.capture_name, self.window_start = name, start
                self.y = type("T", (), {"item": lambda s: 0})()
        graphs = [G(c, float(s)) for c in ("ambient_a", "ambient_v") for s in range(0, 99)]
        train, val = partition_benign(graphs, ["ambient_a"], ["ambient_v"], calib_tail=0.2)
        tail = [g for g in val if g.capture_name == "ambient_a"]
        self.assertEqual(sum(g.capture_name == "ambient_v" for g in val), 99)
        self.assertTrue(tail and train)
        self.assertLessEqual(max(g.window_start for g in train) + 2.0, min(g.window_start for g in tail))


if __name__ == "__main__":
    unittest.main()
