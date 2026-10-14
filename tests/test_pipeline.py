
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
from evaluate import compute_roc_auc, compute_pr_auc
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
