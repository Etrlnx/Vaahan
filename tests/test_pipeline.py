
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
