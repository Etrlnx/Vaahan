from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from scorer import AnomalyScorer, TransitionBaseline, AnomalyScorerConfig


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class DetectionResult:
    capture_name: str
    window_start: float
    anomaly_score: float
    reconstruction_error: float
    temporal_error: float
    structural_error: float
    risk_state: RiskState
    ground_truth_label: int = 0
    excess_error: float = 0.0
    top_anomalous_node_indices: list[int] = field(default_factory=list)
    top_anomalous_node_ids: list[int] = field(default_factory=list)
    node_scores: list[float] = field(default_factory=list)


class ZeroDayDetector:
    def __init__(self, scorer_config: Optional[AnomalyScorerConfig] = None):
        self.scorer_config = scorer_config or AnomalyScorerConfig()
        self.scorer = AnomalyScorer(self.scorer_config)
        self.transition_baseline: Optional[TransitionBaseline] = None
        self.tau_suspicious: float = 0.50
        self.tau_alert: float = 1.00
        self.is_calibrated: bool = False

    def fit_baseline(self, benign_train_graphs: list, vocab_size: int = 106, count_zero: Optional[float] = None):
        if not benign_train_graphs:
            raise ValueError("benign_train_graphs is empty; cannot fit transition baseline.")
        self.transition_baseline = TransitionBaseline(vocab_size=vocab_size)
        self.transition_baseline.fit(benign_train_graphs, count_zero=count_zero)

    def calibrate(
        self,
        model,
        val_loader,
        device=torch.device("cpu"),
        suspicious_percentile: float = 95.0,
        alert_percentile: float = 99.0
    ):
        if not (0.0 < suspicious_percentile < alert_percentile <= 100.0):
            raise ValueError(
                f"Invalid calibration percentiles: require 0 < suspicious_percentile ({suspicious_percentile}) "
                f"< alert_percentile ({alert_percentile}) <= 100."
            )

        val_loader = list(val_loader)
        if len(val_loader) == 0:
            raise ValueError("Validation loader yielded 0 windows; cannot calibrate detector thresholds.")

        needs_baseline = (self.scorer_config.gamma_struct > 0 or self.scorer_config.delta_excess > 0
                          or self.scorer_config.beta_temporal > 0)
        if needs_baseline and (self.transition_baseline is None or not self.transition_baseline.fitted):
            raise RuntimeError(
                "Structural weights (gamma_struct / delta_excess) > 0, but TransitionBaseline has not been fitted. "
                "Call detector.fit_baseline(train_graphs) before calibrate()."
            )

        self.scorer.fit_component_scales(model, val_loader, device, self.transition_baseline)

        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(
                    x_stats=batch.x,
                    id_idx=batch.id_idx,
                    edge_index=batch.edge_index,
                    edge_weight=batch.edge_attr,
                )
                score_dict = self.scorer.score_window(
                    batch, outputs, transition_baseline=self.transition_baseline
                )
                val_scores.append(score_dict["anomaly_score"])

        if len(val_scores) == 0:
            raise ValueError("Validation loader yielded 0 windows; cannot calibrate detector thresholds.")

        val_scores_np = np.array(val_scores, dtype=np.float64)
        self.tau_suspicious = float(np.percentile(val_scores_np, suspicious_percentile))
        self.tau_alert = float(np.percentile(val_scores_np, alert_percentile))

        if self.tau_suspicious > self.tau_alert:
            raise RuntimeError(
                f"Calibrated threshold invariant violated: tau_suspicious ({self.tau_suspicious:.4f}) "
                f"> tau_alert ({self.tau_alert:.4f})."
            )

        self.is_calibrated = True
        print(f"Detector Calibrated on {len(val_scores)} validation windows:")
        print(f"  tau_suspicious ({suspicious_percentile}th percentile) : {self.tau_suspicious:.4f}")
        print(f"  tau_alert      ({alert_percentile}th percentile) : {self.tau_alert:.4f}")

    def evaluate_graph(self, batch, outputs) -> DetectionResult:
        if not self.is_calibrated:
            raise RuntimeError(
                "Detector must be calibrated before evaluating graphs. "
                "Call detector.calibrate(model, val_loader) first."
            )

        score_dict = self.scorer.score_window(
            batch, outputs, transition_baseline=self.transition_baseline
        )
        score = score_dict["anomaly_score"]

        if score >= self.tau_alert:
            risk = RiskState.HIGH_RISK
        elif score >= self.tau_suspicious:
            risk = RiskState.SUSPICIOUS
        else:
            risk = RiskState.NORMAL

        # Rank nodes by the evidence that drives the score: structural excess, or with the rate term
        # on, max(E, T) per node in the same scaled units as the fused score (a retimed ID is blamed
        # for its own rate, not its neighbours' shifted transitions). Fall back to reconstruction
        # residuals when no transition baseline is available.
        node_scores = np.asarray(score_dict["node_excess"], dtype=np.float64)
        cfg = self.scorer_config
        if cfg.beta_temporal > 0 and cfg.excess_scale and cfg.temporal_scale:
            node_scores = np.maximum(cfg.delta_excess * node_scores / cfg.excess_scale,
                                     cfg.beta_temporal * np.asarray(score_dict["node_rate"]) / cfg.temporal_scale)
        if self.transition_baseline is None or not np.any(node_scores):
            node_scores = score_dict["node_residuals"].cpu().numpy()
        top_k = min(3, len(node_scores))
        top_node_idx = np.argsort(node_scores)[::-1][:top_k].tolist()
        top_id_vocab_indices = batch.id_idx[top_node_idx].cpu().numpy().tolist()

        raw_capture = batch.capture_name
        capture_name = raw_capture[0] if isinstance(raw_capture, (list, tuple)) else str(raw_capture)
        window_start = float(batch.window_start.item()) if hasattr(batch.window_start, "item") else float(batch.window_start[0]) if isinstance(batch.window_start, (list, tuple)) else 0.0
        label = int(batch.y[0].item())

        return DetectionResult(
            capture_name=capture_name,
            window_start=window_start,
            anomaly_score=score,
            reconstruction_error=score_dict["r_error"],
            temporal_error=score_dict["t_error"],
            structural_error=score_dict["g_error"],
            risk_state=risk,
            ground_truth_label=label,
            excess_error=score_dict["e_error"],
            top_anomalous_node_indices=top_node_idx,
            top_anomalous_node_ids=top_id_vocab_indices,
            node_scores=node_scores.tolist(),
        )

    def export_calibration(self) -> dict:
        return {
            "tau_suspicious": self.tau_suspicious,
            "tau_alert": self.tau_alert,
            "is_calibrated": self.is_calibrated,
            "scorer_config": self.scorer_config.__dict__,
        }

    def import_calibration(self, state: dict):
        self.tau_suspicious = state["tau_suspicious"]
        self.tau_alert = state["tau_alert"]
        self.is_calibrated = state.get("is_calibrated", True)
