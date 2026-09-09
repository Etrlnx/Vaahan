"""
XAI Ground-Truth Fidelity & Explanation Benchmark
--------------------------------------------------------------------
Evaluates the explainability layer quantitatively against documented
ROAD attack ground-truth targets (CAN Arbitration IDs and physical signals).

Computes:
  - Hit Rate @ Top-1 (HR@1): Top attributed CAN ID matches ground truth target
  - Hit Rate @ Top-3 (HR@3): Ground truth target in Top-3 attributed CAN IDs
  - Feature Grounding Fidelity: Attributed feature matches attack mechanism
  - Sample Security Incident Report generation (JSON + Text)
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")

for p in (GT_DIR, AD_DIR, BASE_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.append(p)

from model import GraphTransformerAutoencoder, ModelConfig
from train import align_graph_feature_dim
from evaluate import restore_or_split_graphs
from detector import ZeroDayDetector, RiskState
from scorer import AnomalyScorerConfig
from report_generator import ReportGenerator, SecurityIncidentReport


OUTPUTS_DIR = os.path.join(GT_DIR, "outputs")
GRAPH_DATA_PATH = os.path.join(OUTPUTS_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUTS_DIR, "best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Documented ground-truth target IDs and signals in ROAD dataset
GROUND_TRUTH_TARGETS = {
    "speedometer": {
        "target_ids": ["0x0D0", "208", "0x0D1", "209", "0x430", "1072", "0x434", "1076", "0x350", "848"],
        "expected_features": ["signal_max", "signal_range", "signal_mean"],
    },
    "reverse_light": {
        "target_ids": ["0x0B4", "180", "0x0B6", "182", "0x2B0", "688", "0x320", "800"],
        "expected_features": ["signal_mean", "signal_max", "signal_min", "signal_range"],
    },
    "correlated": {
        "target_ids": ["0x0D0", "208", "0x0D1", "209", "0x1A0", "416"],
        "expected_features": ["signal_mean", "signal_range", "signal_std"],
    },
    "coolant": {
        "target_ids": ["0x0F2", "242", "0x0C0", "192", "0x1F0", "496"],
        "expected_features": ["signal_max", "signal_mean", "signal_range"],
    },
    "accelerator": {
        "target_ids": ["0x0C0", "192", "0x0D0", "208", "0x350", "848"],
        "expected_features": ["signal_max", "signal_mean", "msg_count", "mean_iat"],
    },
}


def normalize_can_id(id_val) -> set:
    """Returns a set of canonical string representations (hex and decimal) for an ID."""
    id_str = str(id_val).strip()
    res = {id_str.lower(), id_str.upper()}
    try:
        if id_str.startswith(("0x", "0X")):
            dec = int(id_str, 16)
        else:
            dec = int(id_str)
        res.add(str(dec))
        res.add(f"0x{dec:03X}")
        res.add(f"0x{dec:X}")
        res.add(f"0x{dec:03x}")
        res.add(f"0x{dec:x}")
    except ValueError:
        pass
    return res


def ids_match(id_a: str, id_b: str) -> bool:
    return len(normalize_can_id(id_a).intersection(normalize_can_id(id_b))) > 0


def is_target_hit(predicted_id: str, target_id_list: list) -> bool:
    return any(ids_match(predicted_id, tid) for tid in target_id_list)


def get_ground_truth_for_capture(capture_name: str) -> dict:
    name_lower = capture_name.lower()
    for key, gt in GROUND_TRUTH_TARGETS.items():
        if key in name_lower:
            return gt
    return {"target_ids": [], "expected_features": []}


from plot_component_radar import plot_component_radar
from plot_id_attribution import plot_id_attribution
from plot_interactive_subgraph import plot_interactive_subgraph
from plot_xai_dashboard import render_unified_dashboard


def run_xai_evaluation(show_plots: bool = False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 80)
    print("EXPLAINABLE AI (XAI) GROUND-TRUTH FIDELITY BENCHMARK")
    print("=" * 80)

    if not os.path.exists(GRAPH_DATA_PATH) or not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("Missing graph data or checkpoint. Please run train.py first.")

    print("\n[1/3] Loading model, vocabulary, and test graphs...")
    graphs = torch.load(GRAPH_DATA_PATH, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    config = ModelConfig()
    if "config" in checkpoint:
        for k, v in checkpoint["config"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    config.num_ids = len(vocab)
    align_graph_feature_dim(graphs, config.node_stat_feature_dim)

    train_graphs, val_graphs, test_graphs, train_mean, train_std = restore_or_split_graphs(graphs, checkpoint)

    model = GraphTransformerAutoencoder(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scorer_config = AnomalyScorerConfig(alpha_recon=0.50, beta_temporal=0.25, gamma_struct=0.25)
    detector = ZeroDayDetector(scorer_config)
    detector.fit_baseline(train_graphs, vocab_size=len(vocab))

    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    detector.calibrate(model, val_loader, device=DEVICE, suspicious_percentile=95.0, alert_percentile=99.0)

    report_gen = ReportGenerator(vocab=vocab, transition_baseline=detector.transition_baseline)

    print("\n[2/3] Evaluating Explanation Fidelity on True Attack Windows...")
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    total_attack_alerts = 0
    hit_top1_count = 0
    hit_top3_count = 0
    feature_match_count = 0

    sample_reports: list[SecurityIncidentReport] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            res = detector.evaluate_graph(batch, outputs)

            # Evaluate explanations on attack windows flagged as suspicious/alert
            if res.ground_truth_label == 1 and res.anomaly_score >= detector.tau_suspicious:
                total_attack_alerts += 1
                gt = get_ground_truth_for_capture(res.capture_name)
                target_ids = gt["target_ids"]
                expected_feats = gt["expected_features"]

                report = report_gen.generate_report(
                    batch, outputs, res, model,
                    decision_threshold=detector.tau_suspicious,
                    alert_threshold=detector.tau_alert
                )

                # Collect distinct attack incident reports for visualization
                if len(sample_reports) < 4:
                    if not any(r.capture_name == report.capture_name for r in sample_reports):
                        sample_reports.append(report)

                if target_ids and report.top_anomalous_ids:
                    top1_id = report.top_anomalous_ids[0]["arbitration_id"]
                    top3_ids = [n["arbitration_id"] for n in report.top_anomalous_ids[:3]]

                    if is_target_hit(top1_id, target_ids):
                        hit_top1_count += 1
                    if any(is_target_hit(tid, target_ids) for tid in top3_ids):
                        hit_top3_count += 1

                    # Check feature grounding
                    top1_feats = [f["feature"] for f in report.top_anomalous_ids[0]["top_deviating_features"]]
                    if any(f in expected_feats for f in top1_feats):
                        feature_match_count += 1

    hr_1 = (hit_top1_count / max(total_attack_alerts, 1)) * 100.0
    hr_3 = (hit_top3_count / max(total_attack_alerts, 1)) * 100.0
    feat_fidelity = (feature_match_count / max(total_attack_alerts, 1)) * 100.0

    print("\n" + "=" * 80)
    print("EXPLAINABILITY FIDELITY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Evaluated Attack Alert Windows    : {total_attack_alerts}")
    print(f"  Hit Rate @ Top-1 (HR@1)           : {hr_1:.2f}%  (Top-1 CAN ID matches ground-truth)")
    print(f"  Hit Rate @ Top-3 (HR@3)           : {hr_3:.2f}%  (Target in Top-3 CAN IDs)")
    print(f"  Feature Grounding Fidelity Rate   : {feat_fidelity:.2f}%  (Physical signal matches attack mode)")
    print("=" * 80)

    print("\n[3/3] Generating Visual Explanations & Security Incident Reports...")
    visuals_dir = os.path.join(OUTPUTS_DIR, "xai_visuals")

    # Clear old visuals so every run produces a clean, up-to-date set
    if os.path.isdir(visuals_dir):
        import shutil
        shutil.rmtree(visuals_dir)
    os.makedirs(visuals_dir, exist_ok=True)

    for i, rep in enumerate(sample_reports):
        # Use a fixed, human-readable filename prefix so re-runs overwrite cleanly
        prefix = os.path.join(visuals_dir, f"incident_{i + 1:02d}_{rep.capture_name.replace(' ', '_')}")

        print(f"\n--- Incident {i + 1} [{rep.risk_state}] ({rep.capture_name} @ {rep.timestamp}s) ---")
        print(report_gen.to_text(rep))

        # 1. Component Deviation Radar / Spider Chart
        plot_component_radar(rep, save_path=f"{prefix}_radar_chart.png")

        # 2. CAN ID Anomaly Contribution & Root-Cause Bar Chart
        plot_id_attribution(rep, save_path=f"{prefix}_id_attribution.png")

        # 3. Interactive Anomaly-Annotated Subgraph Network (PNG + HTML)
        plot_interactive_subgraph(
            rep,
            png_save_path=f"{prefix}_subgraph.png",
            html_save_path=f"{prefix}_subgraph_interactive.html",
        )

        # 4. Master Unified Matplotlib Dashboard — save PNG, defer show() until the end
        render_unified_dashboard(rep, save_path=f"{prefix}_dashboard.png", show=False)

    if sample_reports:
        sample_json_path = os.path.join(OUTPUTS_DIR, "sample_security_report.json")
        with open(sample_json_path, "w", encoding="utf-8") as f:
            f.write(report_gen.to_json(sample_reports[0]))
        print(f"\nSaved structured JSON report to: {sample_json_path}")
        print(f"Saved visual XAI dashboards to: {visuals_dir}")

    # Show all dashboards in one blocking call — script pauses here until
    # the user closes ALL open Matplotlib windows, then exits cleanly.
    if show_plots and sample_reports:
        import matplotlib.pyplot as plt
        print("\nDisplaying dashboards — close all windows to exit.")
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate XAI and Generate Visual Dashboards")
    parser.add_argument("--show", action="store_true", help="Display Matplotlib GUI windows interactively on screen")
    args = parser.parse_args()
    run_xai_evaluation(show_plots=args.show)

