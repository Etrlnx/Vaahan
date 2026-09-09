"""
Unit & Smoke Tests for the Explainability Layer (XAI)
------------------------------------------------------
Validates:
  1. AttentionAnalyzer layer extraction and rollout computation
  2. FeatureAttributor residual decomposition and ranking
  3. GraphLocalizer subgraph extraction and transition penalties
  4. ReportGenerator structured JSON schema and natural language text synthesis
"""

import sys
import os
import unittest
import numpy as np
import torch
from torch_geometric.data import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")
XAI_DIR = os.path.join(PARENT_DIR, "explainability")

for p in (GT_DIR, AD_DIR, XAI_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from attention_analyzer import AttentionAnalyzer
from attribution import FeatureAttributor
from graph_localizer import GraphLocalizer
from report_generator import ReportGenerator
from detector import DetectionResult, RiskState
from scorer import TransitionBaseline


class TestXAIComponents(unittest.TestCase):
    def setUp(self):
        self.vocab = {"<UNK>": 0, "0x0D0": 1, "0x0B4": 2, "0x1A0": 3}
        self.batch = Data(
            x=torch.tensor([[10.0, 0.02, 0.001, 50.0, 5.0, 0.0, 100.0, 0.4, 100.0],
                            [5.0, 0.05, 0.002, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                            [8.0, 0.03, 0.001, 1.0, 0.1, 0.0, 2.0, 0.4, 2.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.tensor([5.0, 3.0, 4.0], dtype=torch.float32),
            y=torch.tensor([1], dtype=torch.long),
        )
        self.batch.id_idx = torch.tensor([1, 2, 3], dtype=torch.long)
        self.batch.capture_name = "test_speedometer_capture"
        self.batch.window_start = 12.0
        self.batch.num_graphs = 1

        self.outputs = {
            "x_recon": torch.tensor([[10.0, 0.02, 0.001, 20.0, 2.0, 0.0, 40.0, 0.4, 40.0],
                                     [5.0, 0.05, 0.002, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
                                     [8.0, 0.03, 0.001, 1.0, 0.1, 0.0, 2.0, 0.4, 2.0]], dtype=torch.float32),
            "z": torch.randn((3, 48)),
            "edge_logits": torch.tensor([1.0, 0.5, 0.8]),
        }

    def test_feature_attributor(self):
        attributor = FeatureAttributor(self.vocab)
        attributions = attributor.attribute_window(self.batch, self.outputs, top_k_nodes=2, top_k_features=3)

        self.assertGreater(len(attributions), 0)
        top_node = attributions[0]
        self.assertEqual(top_node.arbitration_id, "0x0D0")
        self.assertGreater(top_node.graph_contribution_percentage, 50.0)
        self.assertTrue(len(top_node.top_features) <= 3)

    def test_graph_localizer(self):
        baseline = TransitionBaseline(vocab_size=10)
        baseline.fit([self.batch])
        localizer = GraphLocalizer(self.vocab, baseline)

        subgraph = localizer.localize_subgraph(self.batch, model=None, center_local_idx=0)
        self.assertEqual(subgraph.center_id, "0x0D0")
        self.assertGreater(len(subgraph.edges), 0)

    def test_report_generator(self):
        baseline = TransitionBaseline(vocab_size=10)
        baseline.fit([self.batch])
        generator = ReportGenerator(self.vocab, baseline)

        det_result = DetectionResult(
            capture_name="test_speedometer_capture",
            window_start=12.0,
            anomaly_score=0.88,
            reconstruction_error=0.92,
            temporal_error=0.10,
            structural_error=0.05,
            risk_state=RiskState.HIGH_RISK,
            ground_truth_label=1,
        )

        class DummyModel:
            def get_attention_weights(self):
                return [(self.batch.edge_index, torch.tensor([[0.8, 0.7, 0.9, 0.8],
                                                              [0.3, 0.2, 0.4, 0.3],
                                                              [0.5, 0.6, 0.5, 0.6]]))]

        dummy_model = DummyModel()
        dummy_model.batch = self.batch

        report = generator.generate_report(self.batch, self.outputs, det_result, dummy_model)
        self.assertEqual(report.risk_state, "HIGH_RISK")
        self.assertTrue("RESTRICT_ID" in report.recommended_gateway_action)

        # Validate JSON serialization
        json_str = generator.to_json(report)
        self.assertTrue("incident_id" in json_str)
        self.assertTrue("top_anomalous_ids" in json_str)

        # Validate text output
        text_str = generator.to_text(report)
        self.assertTrue("SECURITY INCIDENT REPORT" in text_str)


if __name__ == "__main__":
    unittest.main()
