import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.append(GT_DIR)


@dataclass
class AnomalyScorerConfig:
    # Fusion weights (sum to 1.0)
    alpha_recon: float = 0.50
    beta_temporal: float = 0.25
    gamma_struct: float = 0.25
    # Node feature importance weights matching the GAE loss
    feature_weights: tuple = (1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5)
    # Structural edge weight multiplier
    edge_loss_weight: float = 0.50
    # Minimum probability threshold for a transition to be considered "normal"
    rare_transition_prob_threshold: float = 1e-4
    # Component scaling (fitted on validation set)
    recon_scale: Optional[float] = None
    temporal_scale: Optional[float] = None
    struct_scale: Optional[float] = None


class TransitionBaseline:
    """
    Builds and maintains empirical transition matrices P(ID_j | ID_i)
    from benign ambient CAN graphs to detect abnormal message sequencing.
    """
    def __init__(self, vocab_size: int = 106):
        self.vocab_size = vocab_size
        self.transition_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.transition_probs = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.fitted = False

    def fit(self, benign_graphs: list):
        """Learns normal message transition probabilities from benign graphs."""
        if not benign_graphs:
            raise ValueError("benign_graphs is empty; cannot fit transition baseline.")
        for g in benign_graphs:
            if not hasattr(g, "edge_index") or g.edge_index.shape[1] == 0:
                continue
            src_nodes = g.edge_index[0].cpu().numpy()
            dst_nodes = g.edge_index[1].cpu().numpy()
            weights = g.edge_attr.cpu().numpy() if hasattr(g, "edge_attr") and g.edge_attr is not None else np.ones(len(src_nodes))
            src_ids = g.id_idx[src_nodes].cpu().numpy()
            dst_ids = g.id_idx[dst_nodes].cpu().numpy()
            for s, d, w in zip(src_ids, dst_ids, weights):
                if s < self.vocab_size and d < self.vocab_size:
                    self.transition_counts[s, d] += float(w)
        # Normalize rows with Laplace smoothing
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        smoothed_counts = self.transition_counts + 1e-3
        smoothed_sums = row_sums + (1e-3 * self.vocab_size)
        self.transition_probs = smoothed_counts / np.maximum(smoothed_sums, 1e-6)
        self.fitted = True

    def compute_structural_penalty(self, g) -> float:
        """
        Computes the negative log-likelihood penalty for observed edges in graph g.
        High penalty indicates unexpected / unseen CAN ID sequences (masquerade signature).
        """
        if not self.fitted or g.edge_index.shape[1] == 0:
            return 0.0
        src_nodes = g.edge_index[0].cpu().numpy()
        dst_nodes = g.edge_index[1].cpu().numpy()
        weights = g.edge_attr.cpu().numpy() if hasattr(g, "edge_attr") and g.edge_attr is not None else np.ones(len(src_nodes))

        src_ids = g.id_idx[src_nodes].cpu().numpy()
        dst_ids = g.id_idx[dst_nodes].cpu().numpy()
        penalties = []
        for s, d, w in zip(src_ids, dst_ids, weights):
            if s < self.vocab_size and d < self.vocab_size:
                prob = self.transition_probs[s, d]
                nll = -np.log(np.maximum(prob, 1e-7))
                penalties.append(nll * float(w))
        if not penalties:
            return 0.0
        total_weight = np.maximum(np.sum(weights), 1.0)
        return float(np.sum(penalties) / total_weight)


class AnomalyScorer:
    """
    Evaluates individual graph windows and computes multi-component anomaly scores.
    """
    # Feature indices (from graph_builder.py extract_node_features):
    # 0: msg_count, 1: mean_iat, 2: std_iat, 3: signal_mean, 4: signal_std,
    # 5: signal_min, 6: signal_max, 7: activity_share, 8: signal_range
    TEMPORAL_FEATURE_IDX = [1, 2, 7]

    def __init__(self, config: AnomalyScorerConfig = None):
        self.config = config or AnomalyScorerConfig()
        self.feat_weights_tensor = torch.tensor(
            self.config.feature_weights, dtype=torch.float32
        )

    def compute_reconstruction_deviation(self, batch, outputs) -> tuple[float, torch.Tensor]:
        device = batch.x.device
        weights = self.feat_weights_tensor.to(device)
        node_residuals = (((outputs["x_recon"] - batch.x) ** 2) * weights).mean(dim=-1)
        node_error = node_residuals.mean().item()
        edge_error = 0.0
        if "edge_logits" in outputs and outputs["edge_logits"] is not None and batch.edge_index.shape[1] > 0:
            pos_labels = torch.ones_like(outputs["edge_logits"])
            edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], pos_labels)
            edge_error = edge_loss.item()
        total_recon_error = node_error + (self.config.edge_loss_weight * edge_error)
        return total_recon_error, node_residuals.detach()

    def compute_temporal_deviation(self, batch, outputs) -> float:
        x_recon = outputs["x_recon"]
        x_true = batch.x
        if x_recon.shape[-1] <= max(self.TEMPORAL_FEATURE_IDX):
            return 0.0
        temporal_residuals = (x_recon[:, self.TEMPORAL_FEATURE_IDX] - x_true[:, self.TEMPORAL_FEATURE_IDX]) ** 2
        temporal_error = temporal_residuals.mean().item()
        return max(0.0, temporal_error)

    def fit_component_scales(self, model, val_loader, device, transition_baseline: Optional[TransitionBaseline] = None):
        model.eval()
        recon_errors = []
        temporal_errors = []
        struct_errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(
                    x_stats=batch.x,
                    id_idx=batch.id_idx,
                    edge_index=batch.edge_index,
                    edge_weight=batch.edge_attr,
                )
                r_error, _ = self.compute_reconstruction_deviation(batch, outputs)
                t_error = self.compute_temporal_deviation(batch, outputs)
                g_error = transition_baseline.compute_structural_penalty(batch) if transition_baseline else 0.0

                recon_errors.append(r_error)
                temporal_errors.append(t_error)
                struct_errors.append(g_error)

        def mad_scale(arr):
            arr = np.array(arr, dtype=np.float64)
            if len(arr) == 0:
                return 1.0
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            return float(1.4826 * max(mad, 1e-6))

        self.config.recon_scale = mad_scale(recon_errors)
        self.config.temporal_scale = mad_scale(temporal_errors)
        self.config.struct_scale = mad_scale(struct_errors)

        print(f"Component scales fitted: recon={self.config.recon_scale:.4f}, "
              f"temporal={self.config.temporal_scale:.4f}, struct={self.config.struct_scale:.4f}")

    def score_window(self, batch, outputs, transition_baseline: TransitionBaseline = None) -> dict:
        r_error, node_residuals = self.compute_reconstruction_deviation(batch, outputs)
        t_error = self.compute_temporal_deviation(batch, outputs)
        g_error = transition_baseline.compute_structural_penalty(batch) if transition_baseline else 0.0

        if self.config.recon_scale is not None:
            r_error /= self.config.recon_scale
        if self.config.temporal_scale is not None:
            t_error /= self.config.temporal_scale
        if self.config.struct_scale is not None:
            g_error /= self.config.struct_scale

        fused_score = (
            self.config.alpha_recon * r_error +
            self.config.beta_temporal * t_error +
            self.config.gamma_struct * g_error
        )
        return {
            "anomaly_score": float(fused_score),
            "r_error": float(r_error),
            "t_error": float(t_error),
            "g_error": float(g_error),
            "node_residuals": node_residuals,
        }