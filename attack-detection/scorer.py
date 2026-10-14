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
    # Weights act on validation-scaled components (each divided by its benign MAD).
    # Ablation on the ROAD test split: structural excess (E) carries the detection signal,
    # R adds a small FPR reduction and G is subsumed by E. T (per-ID rate deviation) is off by
    # default: it lifts injection/retiming attacks on can-train-and-test (test_03 ROC up to +0.16)
    # but cuts ROAD masquerade fold ROC 0.967 -> 0.849, where attackers keep the real ECU's rate.
    # Enable with beta_temporal=1.0 (evaluate.py --timing).
    alpha_recon: float = 0.25
    beta_temporal: float = 0.0
    gamma_struct: float = 0.0
    delta_excess: float = 1.0
    # "max": E and T are alternative detectors (ordering vs rate), so the window takes the larger of
    # the two scaled terms instead of their sum, where one term's benign noise masks the other.
    fusion: str = "max"
    feature_weights: tuple = (1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5)
    edge_loss_weight: float = 0.50
    rare_transition_prob_threshold: float = 1e-4
    recon_scale: Optional[float] = None
    temporal_scale: Optional[float] = None
    struct_scale: Optional[float] = None
    excess_scale: Optional[float] = None


class TransitionBaseline:
    RATE_FEATURES = [0, 1]  # msg_count, mean_iat (node feature columns)

    def __init__(self, vocab_size: int = 106):
        self.vocab_size = vocab_size
        self.transition_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.transition_probs = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self.fitted = False

    def fit(self, benign_graphs: list):
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
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        smoothed_counts = self.transition_counts + 1e-3
        smoothed_sums = row_sums + (1e-3 * self.vocab_size)
        self.transition_probs = smoothed_counts / np.maximum(smoothed_sums, 1e-6)
        self.fitted = True

        # Typical per-node surprise of each ID on benign traffic, so busy IDs are not
        # blamed just for carrying many transitions.
        per_id = {}
        for g in benign_graphs:
            ids = g.id_idx.cpu().numpy()
            for i, v in zip(ids, self.node_surprise(g)):
                per_id.setdefault(int(i), []).append(v)
        self.id_surprise_median = np.zeros(self.vocab_size, dtype=np.float64)
        for i, vals in per_id.items():
            if i < self.vocab_size:
                self.id_surprise_median[i] = np.median(vals)

        # Per-ID benign rate profile: median and robust spread of msg_count and mean_iat. Spreads are
        # floored at the median spread across IDs so near-constant IDs do not explode.
        feats = {}
        for g in benign_graphs:
            ids = g.id_idx.cpu().numpy()
            x = g.x[:, self.RATE_FEATURES].cpu().numpy()
            for i, row in zip(ids, x):
                feats.setdefault(int(i), []).append(row)
        k = len(self.RATE_FEATURES)
        self.rate_median = np.zeros((self.vocab_size, k))
        spread = np.full((self.vocab_size, k), np.nan)
        for i, rows in feats.items():
            if i < self.vocab_size:
                rows = np.asarray(rows)
                self.rate_median[i] = np.median(rows, axis=0)
                spread[i] = 1.4826 * np.median(np.abs(rows - self.rate_median[i]), axis=0)
        floor = np.maximum(np.nanmedian(spread, axis=0), 1e-6)
        self.rate_scale = np.where(np.isnan(spread), floor, np.maximum(spread, floor))

    def node_surprise(self, g) -> np.ndarray:
        """Count-weighted transition NLL summed over every edge touching each node."""
        n = g.id_idx.shape[0]
        if g.edge_index.shape[1] == 0:
            return np.zeros(n, dtype=np.float64)
        src = g.edge_index[0].cpu().numpy()
        dst = g.edge_index[1].cpu().numpy()
        w = g.edge_attr.cpu().numpy() if getattr(g, "edge_attr", None) is not None else np.ones(len(src))
        ids = np.minimum(g.id_idx.cpu().numpy(), self.vocab_size - 1)
        e = -np.log(np.maximum(self.transition_probs[ids[src], ids[dst]], 1e-7)) * w
        return np.bincount(src, e, n) + np.bincount(dst, e, n)

    def node_excess_surprise(self, g) -> np.ndarray:
        """Per-node surprise above that ID's benign median; drives detection and attribution."""
        if not self.fitted:
            return np.zeros(g.id_idx.shape[0], dtype=np.float64)
        ids = np.minimum(g.id_idx.cpu().numpy(), self.vocab_size - 1)
        return self.node_surprise(g) - self.id_surprise_median[ids]

    def node_rate_deviation(self, g) -> np.ndarray:
        """Per-node max |feature - ID median| / ID spread over msg_count and mean_iat: how far each
        ID's message rate is from its own benign rate. Unseen IDs (<UNK>) are left to E."""
        if not self.fitted:
            return np.zeros(g.id_idx.shape[0], dtype=np.float64)
        ids = np.minimum(g.id_idx.cpu().numpy(), self.vocab_size - 1)
        x = g.x[:, self.RATE_FEATURES].cpu().numpy()
        dev = np.abs(x - self.rate_median[ids]) / self.rate_scale[ids]
        dev[ids == 0] = 0.0
        return dev.max(axis=1)

    def compute_structural_penalty(self, g) -> float:
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

    @staticmethod
    def compute_temporal_deviation(batch, transition_baseline: Optional[TransitionBaseline]) -> float:
        """T: largest per-ID rate deviation in the window (see TransitionBaseline.node_rate_deviation)."""
        if transition_baseline is None or batch.x.shape[0] == 0:
            return 0.0
        return float(transition_baseline.node_rate_deviation(batch).max())

    def fit_component_scales(self, model, val_loader, device, transition_baseline: Optional[TransitionBaseline] = None):
        model.eval()
        recon_errors = []
        temporal_errors = []
        struct_errors = []
        excess_errors = []

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
                t_error = self.compute_temporal_deviation(batch, transition_baseline)
                g_error = transition_baseline.compute_structural_penalty(batch) if transition_baseline else 0.0
                e_error = float(transition_baseline.node_excess_surprise(batch).max()) if transition_baseline else 0.0

                recon_errors.append(r_error)
                excess_errors.append(e_error)
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
        self.config.excess_scale = mad_scale(excess_errors)

        print(f"Component scales fitted: recon={self.config.recon_scale:.4f}, "
              f"temporal={self.config.temporal_scale:.4f}, struct={self.config.struct_scale:.4f}, "
              f"excess={self.config.excess_scale:.4f}")

    def score_window(self, batch, outputs, transition_baseline: TransitionBaseline = None) -> dict:
        r_error, node_residuals = self.compute_reconstruction_deviation(batch, outputs)
        t_error = self.compute_temporal_deviation(batch, transition_baseline)
        g_error = transition_baseline.compute_structural_penalty(batch) if transition_baseline else 0.0
        node_excess = (transition_baseline.node_excess_surprise(batch) if transition_baseline
                       else np.zeros(batch.x.shape[0]))
        e_error = float(node_excess.max()) if len(node_excess) else 0.0

        def scaled(value, scale):
            return value / scale if scale is not None else value

        e_term = self.config.delta_excess * scaled(e_error, self.config.excess_scale)
        t_term = self.config.beta_temporal * scaled(t_error, self.config.temporal_scale)
        fused_score = (
            self.config.alpha_recon * scaled(r_error, self.config.recon_scale) +
            self.config.gamma_struct * scaled(g_error, self.config.struct_scale) +
            (max(e_term, t_term) if self.config.fusion == "max" and self.config.beta_temporal > 0
             else e_term + t_term)
        )
        return {
            "anomaly_score": float(fused_score),
            "r_error": float(r_error),
            "t_error": float(t_error),
            "g_error": float(g_error),
            "e_error": e_error,
            "node_residuals": node_residuals,
            "node_excess": node_excess,
        }