
import os
import sys
import json
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
from typing import Optional
from detector import ZeroDayDetector, RiskState, DetectionResult
from scorer import AnomalyScorerConfig
from report_generator import ReportGenerator, SecurityIncidentReport


from preprocess import OUTPUT_DIR as OUTPUTS_DIR
GRAPH_DATA_PATH = os.path.join(OUTPUTS_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUTS_DIR, "best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

ATTACK_METADATA_PATH = os.path.join(GT_DIR, "road", "road", "signal_extractions", "attacks", "metadata.json")

# Expected physical tell per attack family; the attacked ID itself comes from ROAD metadata.
EXPECTED_FEATURES = {
    "speedometer": ["signal_max", "signal_range", "signal_mean"],
    "reverse_light": ["signal_mean", "signal_max", "signal_min", "signal_range"],
    "correlated": ["signal_mean", "signal_range", "signal_std"],
    "coolant": ["signal_max", "signal_mean", "signal_range"],
    # Synthetic timing attacks (attack_injector.py): the tell is cadence / volume, not payload.
    "fabrication": ["msg_count", "mean_iat", "std_iat", "activity_share"],
    "suspension": ["msg_count", "mean_iat", "std_iat", "activity_share"],
    "drift": ["mean_iat", "std_iat", "msg_count"],
    "flooding": ["msg_count", "activity_share", "mean_iat"],
}
SYNTHETIC_METADATA_PATH = os.path.join(GT_DIR, "outputs", "synthetic", "metadata.json")


def load_injection_ids(path: str = ATTACK_METADATA_PATH) -> dict:
    """Map capture name -> injected arbitration ID, verified against per-message Label==1 rows."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {name: m["injection_id"] for name, m in meta.items() if m.get("injection_id")}


INJECTION_IDS = load_injection_ids()
if os.path.exists(SYNTHETIC_METADATA_PATH):
    with open(SYNTHETIC_METADATA_PATH, "r", encoding="utf-8") as _f:
        INJECTION_IDS.update({name: m["target_id"] for name, m in json.load(_f).items()})


def normalize_can_id(id_val) -> set:
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
    name = capture_name.removesuffix("_masquerade")
    target = INJECTION_IDS.get(capture_name) or INJECTION_IDS.get(name)
    family = name.split("__")[2] if name.startswith("synthetic__") else name  # skip parent name
    feats = next((v for k, v in EXPECTED_FEATURES.items() if k in family.lower()), [])
    return {"target_ids": [target] if target else [], "expected_features": feats}


def get_attack_family(capture_name: str) -> str:
    c = str(capture_name).lower()
    if "speedometer" in c:
        return "speedometer"
    if "reverse_light_off" in c:
        return "reverse_light_off"
    if "reverse_light_on" in c:
        return "reverse_light_on"
    if "correlated" in c:
        return "correlated_signal"
    if "coolant" in c:
        return "coolant"
    if "accelerator" in c:
        return "accelerator"
    return c


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

    detector = ZeroDayDetector(AnomalyScorerConfig())
    detector.fit_baseline(train_graphs, vocab_size=len(vocab))

    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    detector.calibrate(model, val_loader, device=DEVICE, suspicious_percentile=95.0, alert_percentile=99.0)

    report_gen = ReportGenerator(vocab=vocab, transition_baseline=detector.transition_baseline)

    print("\n[2/3] Evaluating Explanation Fidelity on True Attack Windows...")
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # Fidelity counters kept apart for real ROAD and synthetic attacks.
    counts = {grp: {"alerts": 0, "top1": 0, "top3": 0, "feat": 0} for grp in ("road", "synthetic")}

    attack_candidates: dict[str, tuple[DetectionResult, SecurityIncidentReport]] = {}
    benign_report: Optional[SecurityIncidentReport] = None

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

            if res.ground_truth_label == 0 and res.anomaly_score < detector.tau_suspicious and benign_report is None:
                benign_report = report_gen.generate_report(
                    batch, outputs, res, model,
                    decision_threshold=detector.tau_suspicious,
                    alert_threshold=detector.tau_alert
                )

            is_synthetic = str(res.capture_name).startswith("synthetic__")
            if res.ground_truth_label == 1:
                fam = get_attack_family(res.capture_name)
                if not is_synthetic and (fam not in attack_candidates or res.anomaly_score > attack_candidates[fam][0].anomaly_score):
                    rep = report_gen.generate_report(
                        batch, outputs, res, model,
                        decision_threshold=detector.tau_suspicious,
                        alert_threshold=detector.tau_alert
                    )
                    attack_candidates[fam] = (res, rep)

                if res.anomaly_score >= detector.tau_suspicious:
                    c = counts["synthetic" if is_synthetic else "road"]
                    c["alerts"] += 1
                    gt = get_ground_truth_for_capture(res.capture_name)
                    target_ids = gt["target_ids"]
                    expected_feats = gt["expected_features"]

                    report = report_gen.generate_report(
                        batch, outputs, res, model,
                        decision_threshold=detector.tau_suspicious,
                        alert_threshold=detector.tau_alert
                    )

                    if target_ids and report.top_anomalous_ids:
                        top1_id = report.top_anomalous_ids[0]["arbitration_id"]
                        top3_ids = [n["arbitration_id"] for n in report.top_anomalous_ids[:3]]

                        if is_target_hit(top1_id, target_ids):
                            c["top1"] += 1
                        if any(is_target_hit(tid, target_ids) for tid in top3_ids):
                            c["top3"] += 1

                        top1_feats = [f["feature"] for f in report.top_anomalous_ids[0]["top_deviating_features"]]
                        if any(f in expected_feats for f in top1_feats):
                            c["feat"] += 1

    sample_reports: list[SecurityIncidentReport] = []
    priority_fams = ["speedometer", "reverse_light_off", "reverse_light_on", "correlated_signal"]
    for fam in priority_fams:
        if fam in attack_candidates:
            sample_reports.append(attack_candidates[fam][1])

    for fam, (_, rep) in attack_candidates.items():
        if len(sample_reports) >= 4:
            break
        if rep not in sample_reports:
            sample_reports.append(rep)


    if benign_report is not None:
        sample_reports.append(benign_report)

    print("\n" + "=" * 80)
    print("EXPLAINABILITY FIDELITY BENCHMARK RESULTS")
    print("=" * 80)
    for grp, c in counts.items():
        n = max(c["alerts"], 1)
        print(f"  [{grp.upper()}] Evaluated Attack Alert Windows : {c['alerts']}")
        print(f"  [{grp.upper()}] Hit Rate @ Top-1 (HR@1)        : {100.0 * c['top1'] / n:.2f}%")
        print(f"  [{grp.upper()}] Hit Rate @ Top-3 (HR@3)        : {100.0 * c['top3'] / n:.2f}%")
        print(f"  [{grp.upper()}] Feature Grounding Fidelity     : {100.0 * c['feat'] / n:.2f}%")
    print("=" * 80)

    print("\n[3/3] Generating Visual Explanations & Security Incident Reports...")
    visuals_dir = os.path.join(OUTPUTS_DIR, "xai_visuals")

    if os.path.isdir(visuals_dir):
        import shutil
        shutil.rmtree(visuals_dir)
    os.makedirs(visuals_dir, exist_ok=True)

    for i, rep in enumerate(sample_reports):
        tag = f"benign_baseline_{rep.capture_name.replace(' ', '_')}" if rep.risk_state == "NORMAL" else rep.capture_name.replace(' ', '_')
        prefix = os.path.join(visuals_dir, f"incident_{i + 1:02d}_{tag}")

        print(f"\n--- Incident {i + 1} [{rep.risk_state}] ({rep.capture_name} @ {rep.timestamp}s) ---")
        print(report_gen.to_text(rep))

        plot_component_radar(rep, save_path=f"{prefix}_radar_chart.png")

        plot_id_attribution(rep, save_path=f"{prefix}_id_attribution.png")

        plot_interactive_subgraph(
            rep,
            png_save_path=f"{prefix}_subgraph.png",
            html_save_path=f"{prefix}_subgraph_interactive.html",
        )

        render_unified_dashboard(rep, save_path=f"{prefix}_dashboard.png", show=False)

    if sample_reports:
        sample_json_path = os.path.join(OUTPUTS_DIR, "sample_security_report.json")
        with open(sample_json_path, "w", encoding="utf-8") as f:
            f.write(report_gen.to_json(sample_reports[0]))
        print(f"\nSaved structured JSON report to: {sample_json_path}")
        print(f"Saved visual XAI dashboards to: {visuals_dir}")

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

