
import sys
import os
import unittest
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")

for p in (GT_DIR, AD_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from plot_confusion_matrix import plot_confusion_matrix
from evaluate import compute_roc_auc, compute_pr_auc, intensity_and_episodes, alarm_events_per_hour
from detector import ZeroDayDetector, RiskState
from scorer import AnomalyScorer, AnomalyScorerConfig, TransitionBaseline


class TestMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        roc_auc = compute_roc_auc(scores, labels)
        self.assertAlmostEqual(roc_auc, 1.0, places=5)

    def test_inverted_ranking(self):
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([0, 0, 0, 1, 1, 1])
        roc_auc = compute_roc_auc(scores, labels)
        self.assertAlmostEqual(roc_auc, 0.0, places=5)

    def test_tied_scores(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 0, 1, 1])
        roc_auc = compute_roc_auc(scores, labels)
        self.assertAlmostEqual(roc_auc, 0.5, places=5)

    def test_single_class_edge_case(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        roc_auc = compute_roc_auc(scores, labels)
        self.assertEqual(roc_auc, 0.5)

    def test_pr_auc_perfect(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        pr_auc = compute_pr_auc(scores, labels)
        self.assertGreaterEqual(pr_auc, 0.90)


class TestDetectorContracts(unittest.TestCase):
    def test_uncalibrated_readiness(self):
        detector = ZeroDayDetector()
        self.assertFalse(detector.is_calibrated)

    def test_invalid_percentiles(self):
        detector = ZeroDayDetector()
        dummy_loader = []
        dummy_model = torch.nn.Identity()
        with self.assertRaises(ValueError):
            detector.calibrate(dummy_model, dummy_loader, suspicious_percentile=99.0, alert_percentile=95.0)

    def test_empty_val_loader(self):
        detector = ZeroDayDetector(AnomalyScorerConfig(gamma_struct=0.0))
        dummy_loader = []
        dummy_model = torch.nn.Identity()
        with self.assertRaises(ValueError):
            detector.calibrate(dummy_model, dummy_loader, suspicious_percentile=90.0, alert_percentile=95.0)

    def test_transition_baseline_smoothing(self):
        baseline = TransitionBaseline(vocab_size=10)
        from torch_geometric.data import Data
        g = Data(
            x=torch.zeros((4, 9)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            edge_attr=torch.tensor([1.0, 2.0, 1.0]),
        )
        g.id_idx = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        baseline.fit([g])
        self.assertTrue(baseline.fitted)
        penalty = baseline.compute_structural_penalty(g)
        self.assertGreater(penalty, 0.0)

    def test_node_excess_flags_injected_id(self):
        from torch_geometric.data import Data

        def chain(ids, counts):
            g = Data(
                x=torch.zeros((len(ids), 9)),
                edge_index=torch.tensor([list(range(len(ids) - 1)), list(range(1, len(ids)))], dtype=torch.long),
                edge_attr=torch.tensor(counts, dtype=torch.float32),
            )
            g.id_idx = torch.tensor(ids, dtype=torch.long)
            return g

        baseline = TransitionBaseline(vocab_size=10)
        baseline.fit([chain([1, 2, 3, 4], [5.0, 5.0, 5.0])] * 3)
        # ID 5 never seen in benign order and inserted between 2 and 3.
        attacked = chain([1, 2, 5, 3, 4], [5.0, 5.0, 5.0, 5.0])
        excess = baseline.node_excess_surprise(attacked)
        self.assertEqual(int(np.argmax(excess)), 2)
        self.assertLessEqual(baseline.node_excess_surprise(chain([1, 2, 3, 4], [5.0, 5.0, 5.0])).max(), 1e-9)

    def test_rate_deviation_flags_changed_period(self):
        from torch_geometric.data import Data
        rng = np.random.default_rng(0)

        def window(count_id2):
            x = torch.zeros((3, 9))
            x[:, 0] = torch.tensor([10.0, count_id2, 10.0]) + torch.tensor(rng.normal(0, 0.5, 3), dtype=torch.float32)
            g = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.ones(2))
            g.id_idx = torch.tensor([1, 2, 3])
            return g

        baseline = TransitionBaseline(vocab_size=5)
        baseline.fit([window(10.0) for _ in range(50)])
        dev = baseline.node_rate_deviation(window(20.0))  # ID 2 now sends twice as often
        self.assertEqual(int(np.argmax(dev)), 1)
        self.assertGreater(dev[1], 5 * max(dev[0], dev[2]))


class TestRateScaleFloor(unittest.TestCase):
    def test_periodic_ids_with_zero_spread_stay_in_message_units(self):
        from torch_geometric.data import Data

        def window(counts):
            x = torch.zeros((3, 9))
            x[:, 0] = torch.tensor(counts, dtype=torch.float32)
            g = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.ones(2))
            g.id_idx = torch.tensor([1, 2, 3])
            return g

        baseline = TransitionBaseline(vocab_size=5)
        baseline.fit([window([10.0, 11.0, 12.0])] * 20)  # every ID perfectly periodic: MAD 0
        dev = baseline.node_rate_deviation(window([12.0, 11.0, 12.0]))
        self.assertAlmostEqual(float(dev[0]), 2.0, places=4)  # two extra messages -> T = 2, not 2e6


class TestIntensityAndEpisodes(unittest.TestCase):
    def test_bins_and_episodes(self):
        from types import SimpleNamespace as NS

        def pair(cap, t, frames, score, label=1):
            g = NS(frac_attack_messages=frames / 100.0, graph_message_count=100.0)
            return g, NS(capture_name=cap, window_start=t, anomaly_score=score, ground_truth_label=label)

        pairs = [pair("a", 0, 1, 0.0), pair("a", 1, 1, 2.0),      # episode 1: caught at t=1
                 pair("a", 5, 50, 0.0),                            # episode 2: missed
                 pair("b", 3, 200, 2.0), pair("b", 9, 0, 9.0, 0)]  # episode 3: caught; benign ignored
        d = intensity_and_episodes(pairs, tau=1.0)
        self.assertEqual((d["episodes"], d["episodes_detected"]), (3, 2))
        self.assertEqual(d["median_latency_s"], 0.5)
        self.assertEqual(d["intensity_recall"]["1"], {"windows": 2, "recall": 0.5})
        self.assertEqual(d["intensity_recall"]["21-100"]["recall"], 0.0)
        self.assertEqual(d["intensity_recall"]["101+"]["recall"], 1.0)


class TestAlarmEvents(unittest.TestCase):
    def test_bursts_count_once(self):
        from types import SimpleNamespace as NS
        rs = [NS(capture_name="a", window_start=float(t), anomaly_score=s)
              for t, s in [(0, 2), (1, 2), (2, 2), (3, 0), (4, 2), (5, 0)]]  # 2 events in 6 s
        rs += [NS(capture_name="b", window_start=float(t), anomaly_score=0.0) for t in range(6)]
        self.assertAlmostEqual(alarm_events_per_hour(rs, tau=1.0), 3600.0 * 2 / 12)


class TestSignalScaling(unittest.TestCase):
    def test_symmetric_log1p(self):
        large_registers = np.array([0.0, 100.0, 1e6, -1e6, 4.32e12])
        scaled = np.sign(large_registers) * np.log1p(np.abs(large_registers))
        self.assertTrue(np.all(np.isfinite(scaled)))
        self.assertLess(np.max(scaled), 50.0)
        self.assertGreater(np.min(scaled), -50.0)

    def test_confusion_matrix_plot_runs_without_crashing(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        fig = plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=["Benign", "Attack"],
            metrics={"roc_auc": 0.75, "pr_auc": 0.80},
            title="Test Confusion Matrix",
            save_path=None,
            show=False,
        )

        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
