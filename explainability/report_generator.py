"""
Security Incident Report Generator (XAI Evidence Fusion)
--------------------------------------------------------------------
Synthesizes multi-source model evidence (attention weights, feature attribution,
reconstruction error, transition structure, and calibrated risk state) into:
  1. Standardized machine-readable JSON schemas (for Gateway Policy).
  2. Natural language security triage reports (for human security analysts).
"""

import os
import sys
import json
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")

for p in (GT_DIR, AD_DIR, BASE_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.append(p)

from detector import DetectionResult, RiskState
from attention_analyzer import AttentionAnalyzer, AttendedEdge
from attribution import FeatureAttributor, NodeAttribution
from graph_localizer import GraphLocalizer, LocalSubgraph
from scorer import TransitionBaseline


# Known functional descriptions for common CAN arbitration IDs in automotive benchmarks
FUNCTIONAL_ID_HINTS = {
    "0x0D0": "Speedometer / Vehicle Wheel Speed",
    "208": "Speedometer / Vehicle Wheel Speed",
    "0x0D1": "Wheel Speed Sensors (FL/FR/RL/RR)",
    "209": "Wheel Speed Sensors (FL/FR/RL/RR)",
    "0x0B4": "Reverse Light / Gear Shift Status",
    "180": "Reverse Light / Gear Shift Status",
    "0x0B6": "Turn Signals & Hazard Indicators",
    "182": "Turn Signals & Hazard Indicators",
    "0x0C0": "Engine RPM / Accelerator Pedal Angle",
    "192": "Engine RPM / Accelerator Pedal Angle",
    "0x0F2": "Engine Coolant Temperature & Thermal Subsystem",
    "242": "Engine Coolant Temperature & Thermal Subsystem",
    "0x1A0": "Steering Angle Sensor & Yaw Rate",
    "416": "Steering Angle Sensor & Yaw Rate",
    "0x2B0": "Brake Pressure & ABS Actuator",
    "688": "Brake Pressure & ABS Actuator",
    "0x320": "Door Lock & Body Control Module",
    "800": "Door Lock & Body Control Module",
    "0x350": "Transmission Control Module (TCM)",
    "848": "Transmission Control Module (TCM)",
    "0x430": "Instrument Cluster & Dashboard Display",
    "1072": "Instrument Cluster & Dashboard Display",
    "0x434": "Instrument Cluster Indicators",
    "1076": "Instrument Cluster Indicators",
}


def lookup_functional_hint(id_val: str) -> str:
    """Matches functional description regardless of whether ID is formatted as hex (0x0D0) or decimal (208)."""
    id_str = str(id_val).strip()
    if id_str in FUNCTIONAL_ID_HINTS:
        return FUNCTIONAL_ID_HINTS[id_str]
    try:
        if id_str.startswith(("0x", "0X")):
            dec_val = int(id_str, 16)
        else:
            dec_val = int(id_str)
        hex_key = f"0x{dec_val:03X}"
        dec_key = str(dec_val)
        return FUNCTIONAL_ID_HINTS.get(hex_key, FUNCTIONAL_ID_HINTS.get(dec_key, "Standard ECU Telemetry"))
    except ValueError:
        return "Standard ECU Telemetry"


@dataclass
class SecurityIncidentReport:
    incident_id: str
    capture_name: str
    timestamp: float
    risk_state: str
    anomaly_score: float
    decision_threshold: float
    alert_threshold: float
    component_breakdown: dict
    top_anomalous_ids: list[dict]
    structural_anomalies: list[dict]
    top_attended_interactions: list[dict]
    recommended_gateway_action: str


class ReportGenerator:
    """
    Fuses all XAI evidence into structured JSON and natural language reports.
    """
    def __init__(
        self,
        vocab: Optional[dict] = None,
        transition_baseline: Optional[TransitionBaseline] = None,
    ):
        self.vocab = vocab or {}
        self.transition_baseline = transition_baseline
        self.attention_analyzer = AttentionAnalyzer(self.vocab)
        self.attributor = FeatureAttributor(self.vocab)
        self.localizer = GraphLocalizer(self.vocab, self.transition_baseline)

    def set_vocab(self, vocab: dict):
        self.vocab = vocab
        self.attention_analyzer.set_vocab(vocab)
        self.attributor.set_vocab(vocab)
        self.localizer.set_vocab(vocab)

    def generate_report(
        self,
        batch,
        outputs: dict,
        detection_result: DetectionResult,
        model,
        decision_threshold: float = 0.7072,
        alert_threshold: float = 1.2440,
    ) -> SecurityIncidentReport:
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"

        # 1. Feature Attribution
        node_attributions = self.attributor.attribute_window(
            batch, outputs, top_k_nodes=3, top_k_features=3
        )

        top_ids_payload = []
        for na in node_attributions:
            func_hint = lookup_functional_hint(na.arbitration_id)
            top_ids_payload.append({
                "arbitration_id": na.arbitration_id,
                "vocab_index": na.vocab_index,
                "functional_description": func_hint,
                "anomaly_contribution_pct": round(na.graph_contribution_percentage, 2),
                "total_node_residual": round(na.total_node_error, 4),
                "top_deviating_features": [
                    {
                        "feature": fa.feature_name,
                        "description": fa.description,
                        "observed_value": round(fa.raw_feature_value, 4),
                        "residual": round(fa.reconstruction_residual, 4),
                        "feature_contribution_pct": round(fa.contribution_percentage, 2),
                    }
                    for fa in na.top_features
                ],
            })

        # 2. Attention Analysis
        attended_edges = self.attention_analyzer.compute_edge_importance(
            batch, model, top_k=4
        )
        attention_payload = [
            {
                "source_id": ae.source_id_str,
                "target_id": ae.target_id_str,
                "attention_weight": round(ae.attention_score, 4),
            }
            for ae in attended_edges
        ]

        # 3. Subgraph Localization & Structural Sequencing
        structural_payload = []
        if node_attributions:
            top_node_local = node_attributions[0].local_index
            subgraph = self.localizer.localize_subgraph(
                batch, model, center_local_idx=top_node_local,
                attention_analyzer=self.attention_analyzer
            )
            for edge in subgraph.edges:
                if edge.is_rare_transition or edge.transition_nll > 5.0:
                    structural_payload.append({
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "transition_count": edge.transition_count,
                        "transition_nll": round(edge.transition_nll, 2),
                        "is_rare_sequence": edge.is_rare_transition,
                    })

        # 4. Gateway Policy Recommendation Rule
        risk = detection_result.risk_state
        if risk == RiskState.HIGH_RISK:
            primary_id = node_attributions[0].arbitration_id if node_attributions else "ALL"
            action = f"RESTRICT_ID ({primary_id}) + ESCALATE_ALERT"
        elif risk == RiskState.SUSPICIOUS:
            action = "HEIGHTENED_MONITORING + LOG_TELEMETRY"
        else:
            action = "ALLOW_FORWARDING"

        return SecurityIncidentReport(
            incident_id=incident_id,
            capture_name=detection_result.capture_name,
            timestamp=round(detection_result.window_start, 2),
            risk_state=str(detection_result.risk_state.value),
            anomaly_score=round(detection_result.anomaly_score, 4),
            decision_threshold=round(decision_threshold, 4),
            alert_threshold=round(alert_threshold, 4),
            component_breakdown={
                "reconstruction_error": round(detection_result.reconstruction_error, 4),
                "temporal_error": round(detection_result.temporal_error, 4),
                "structural_error": round(detection_result.structural_error, 4),
            },
            top_anomalous_ids=top_ids_payload,
            structural_anomalies=structural_payload,
            top_attended_interactions=attention_payload,
            recommended_gateway_action=action,
        )

    def to_json(self, report: SecurityIncidentReport, indent: int = 2) -> str:
        return json.dumps(asdict(report), indent=indent)

    def to_text(self, report: SecurityIncidentReport) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"SECURITY INCIDENT REPORT [{report.risk_state}] — Incident #{report.incident_id}")
        lines.append("=" * 80)
        lines.append(f"Capture: {report.capture_name} | Timestamp: {report.timestamp:.2f}s")
        lines.append(f"Anomaly Score: {report.anomaly_score:.4f} (Decision Tau: {report.decision_threshold:.4f}, Alert Tau: {report.alert_threshold:.4f})")
        lines.append("-" * 80)
        lines.append("COMPONENT ANOMALY BREAKDOWN:")
        lines.append(f"  - Reconstruction Residual (R_error): {report.component_breakdown['reconstruction_error']:.4f}")
        lines.append(f"  - Temporal Dynamics Jitter (T_error): {report.component_breakdown['temporal_error']:.4f}")
        lines.append(f"  - Structural Sequence Penalty (G_error): {report.component_breakdown['structural_error']:.4f}")
        lines.append("-" * 80)
        lines.append("PRIMARY ATTRIBUTED CAN ARBITRATION IDS:")

        for i, node_data in enumerate(report.top_anomalous_ids, start=1):
            lines.append(f"  {i}. CAN ID {node_data['arbitration_id']} ({node_data['functional_description']}):")
            lines.append(f"     -> Graph Anomaly Contribution: {node_data['anomaly_contribution_pct']:.1f}%")
            lines.append(f"     -> Key Deviating Dimensions:")
            for feat in node_data["top_deviating_features"]:
                lines.append(f"        * {feat['feature']:<15} [{feat['description']}]: residual={feat['residual']:.4f} ({feat['feature_contribution_pct']:.1f}%)")

        if report.structural_anomalies:
            lines.append("-" * 80)
            lines.append("STRUCTURAL SEQUENCING ANOMALIES:")
            for sa in report.structural_anomalies:
                rare_tag = " [RARE TRANSITION]" if sa["is_rare_sequence"] else ""
                lines.append(f"  * Transition {sa['source_id']} -> {sa['target_id']} (Count: {sa['transition_count']:.0f}, NLL: {sa['transition_nll']:.2f}){rare_tag}")

        if report.top_attended_interactions:
            lines.append("-" * 80)
            lines.append("TRANSFORMER ATTENTION FOCUS:")
            for ta in report.top_attended_interactions:
                lines.append(f"  * {ta['source_id']:<8} <---> {ta['target_id']:<8} (Attention Weight: {ta['attention_weight']:.4f})")

        lines.append("=" * 80)
        lines.append(f"RECOMMENDED GATEWAY POLICY ACTION: {report.recommended_gateway_action}")
        lines.append("=" * 80)

        return "\n".join(lines)
