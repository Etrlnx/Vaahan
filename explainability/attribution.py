
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")

for p in (GT_DIR, AD_DIR, BASE_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.append(p)


FEATURE_NAMES = [
    "msg_count",
    "mean_iat",
    "std_iat",
    "signal_mean",
    "signal_std",
    "signal_min",
    "signal_max",
    "activity_share",
    "signal_range",
]

FEATURE_DESCRIPTIONS = {
    "msg_count": "Message Volume / Frequency",
    "mean_iat": "Mean Inter-Arrival Time (Transmission Cadence)",
    "std_iat": "Inter-Arrival Time Jitter (Timing Irregularity)",
    "signal_mean": "Decoded Signal Mean Value",
    "signal_std": "Decoded Signal Standard Deviation",
    "signal_min": "Decoded Signal Minimum Bound",
    "signal_max": "Decoded Signal Maximum Bound (Peak Overrides)",
    "activity_share": "Bus Utilization Dominance Share",
    "signal_range": "Dynamic Signal Span Range",
}


@dataclass
class FeatureAttribution:
    feature_name: str
    description: str
    feature_index: int
    raw_feature_value: float
    reconstruction_residual: float
    contribution_percentage: float


@dataclass
class NodeAttribution:
    local_index: int
    vocab_index: int
    arbitration_id: str
    total_node_error: float
    graph_contribution_percentage: float
    top_features: list[FeatureAttribution] = field(default_factory=list)


class FeatureAttributor:
    def __init__(self, vocab: Optional[dict] = None, feature_weights: tuple = (1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5)):
        self.vocab = vocab or {}
        self.idx_to_id = {v: k for k, v in self.vocab.items()} if self.vocab else {}
        self.feature_weights = np.array(feature_weights, dtype=np.float32)

    def set_vocab(self, vocab: dict):
        self.vocab = vocab
        self.idx_to_id = {v: k for k, v in vocab.items()}

    def get_id_str(self, vocab_idx: int) -> str:
        return self.idx_to_id.get(vocab_idx, f"ID_{vocab_idx}")

    def attribute_window(
        self,
        batch,
        outputs: dict,
        top_k_nodes: int = 3,
        top_k_features: int = 3,
    ) -> list[NodeAttribution]:
        x_observed = batch.x.cpu().numpy()
        x_reconstructed = outputs["x_recon"].cpu().numpy()
        global_ids = batch.id_idx.cpu().numpy()
        num_nodes = x_observed.shape[0]

        if num_nodes == 0:
            return []

        squared_diff = (x_reconstructed - x_observed) ** 2
        weighted_residuals = squared_diff * self.feature_weights[np.newaxis, :]

        node_errors = weighted_residuals.sum(axis=-1)
        total_graph_error = max(float(node_errors.sum()), 1e-6)

        ranked_node_indices = np.argsort(node_errors)[::-1]
        results = []

        for rank, u in enumerate(ranked_node_indices[:top_k_nodes]):
            vocab_idx = int(global_ids[u])
            id_str = self.get_id_str(vocab_idx)
            node_err = float(node_errors[u])
            node_contrib_pct = float((node_err / total_graph_error) * 100.0)

            node_residuals = weighted_residuals[u]
            total_node_res = max(float(node_residuals.sum()), 1e-6)

            ranked_feat_indices = np.argsort(node_residuals)[::-1]
            top_feats = []

            for f_idx in ranked_feat_indices[:top_k_features]:
                f_name = FEATURE_NAMES[f_idx]
                f_desc = FEATURE_DESCRIPTIONS.get(f_name, f_name)
                f_val = float(x_observed[u, f_idx])
                f_res = float(node_residuals[f_idx])
                f_pct = float((f_res / total_node_res) * 100.0)

                top_feats.append(FeatureAttribution(
                    feature_name=f_name,
                    description=f_desc,
                    feature_index=int(f_idx),
                    raw_feature_value=f_val,
                    reconstruction_residual=f_res,
                    contribution_percentage=f_pct,
                ))

            results.append(NodeAttribution(
                local_index=int(u),
                vocab_index=vocab_idx,
                arbitration_id=id_str,
                total_node_error=node_err,
                graph_contribution_percentage=node_contrib_pct,
                top_features=top_feats,
            ))

        return results
