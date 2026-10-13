import os
import sys
import pickle
import random
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
if GT_DIR not in sys.path:
    sys.path.append(GT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from model import GraphTransformerAutoencoder, ModelConfig
from train import (split_data, partition_benign, normalize_graph_features, align_graph_feature_dim, apply_saved_normalization,
                   is_test_graph, synthetic_parent, fold_checkpoint_path)
from detector import ZeroDayDetector, RiskState, DetectionResult
from scorer import AnomalyScorerConfig
from plot_confusion_matrix import plot_confusion_matrix
from plot_scenario_explainer import plot_scenario_explainer
from generate_confusion_explainer import generate_confusion_explainer_html

from preprocess import OUTPUT_DIR as OUTPUTS_DIR, DATASET
GRAPH_DATA_PATH = os.path.join(OUTPUTS_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUTS_DIR, "best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def _rank_data_numpy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    n = len(a)
    if n == 0:
        return np.array([], dtype=np.float64)
    sorter = np.argsort(a)
    inv = np.empty(n, dtype=np.intp)
    inv[sorter] = np.arange(n)

    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]

    count = np.r_[np.nonzero(obs)[0], n]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    ranks = _rank_data_numpy(all_scores)
    sum_pos_ranks = np.sum(ranks[all_labels == 1])
    u = sum_pos_ranks - (n_pos * (n_pos + 1.0)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))
    if pos_count == 0 or neg_count == 0:
        return 0.0

    desc_order = np.argsort(scores)[::-1]
    sorted_scores = scores[desc_order]
    sorted_labels = labels[desc_order]

    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, sorted_labels.size - 1]

    tps = np.cumsum(sorted_labels == 1)[threshold_indices]
    fps = np.cumsum(sorted_labels == 0)[threshold_indices]

    precisions = tps / (tps + fps)
    recalls = tps / pos_count

    recalls = np.r_[0.0, recalls]
    precisions = np.r_[precisions[0] if len(precisions) > 0 else 1.0, precisions]

    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap_fn(precisions, recalls))


def restore_or_split_graphs(graphs: list, checkpoint: dict):
    train_captures = checkpoint.get("train_captures")
    val_captures = checkpoint.get("val_captures")
    test_captures = checkpoint.get("test_captures")
    train_mean = checkpoint.get("train_mean")
    train_std = checkpoint.get("train_std")

    if train_captures and val_captures and test_captures and train_mean is not None and train_std is not None:
        print("Restoring exact capture splits and feature normalizer from checkpoint metadata...")
        train_graphs, val_graphs = partition_benign(graphs, train_captures, val_captures,
                                                    checkpoint.get("calib_tail", 0.0))
        test_graphs = [g for g in graphs if is_test_graph(g, test_captures)]
        apply_saved_normalization(train_graphs, train_mean, train_std)
        apply_saved_normalization(val_graphs, train_mean, train_std)
        apply_saved_normalization(test_graphs, train_mean, train_std)
        return train_graphs, val_graphs, test_graphs, train_mean, train_std
    else:
        print("Checkpoint lacks split metadata; falling back to deterministic split...")
        train_graphs, val_graphs, test_graphs, _, _, _ = split_data(graphs)
        mean, std = normalize_graph_features(train_graphs, val_graphs, test_graphs)
        return train_graphs, val_graphs, test_graphs, mean, std


def _binary_metrics(scores, labels, tau):
    preds = (scores >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-6),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "roc_auc": compute_roc_auc(scores, labels),
        "pr_auc": compute_pr_auc(scores, labels),
    }


def summarize_synthetic(results: list, tau: float) -> dict:
    """Per attack type / intensity: window recall and episode detection (one episode per capture)."""
    groups = {}
    for r in results:
        if synthetic_parent(r.capture_name) is None:
            continue
        attack_level = str(r.capture_name).split("__")[2]
        groups.setdefault(attack_level, {}).setdefault(r.capture_name, []).append(r)
    summary = {}
    for key in sorted(groups):
        episodes = list(groups[key].values())
        windows = [r for ep in episodes for r in ep]
        hits = [r for r in windows if r.anomaly_score >= tau]
        latencies = []
        for ep in episodes:
            first = min(r.window_start for r in ep)
            alerts = [r.window_start for r in ep if r.anomaly_score >= tau]
            if alerts:
                latencies.append(min(alerts) - first)
        summary[key] = {
            "windows": len(windows),
            "window_recall": len(hits) / max(len(windows), 1),
            "episodes": len(episodes),
            "episodes_detected": len(latencies),
            "median_latency_s": float(np.median(latencies)) if latencies else None,
        }
    return summary


def run_evaluation(scorer_config: AnomalyScorerConfig = None, show_plot: bool = True,
                   checkpoint_path: str = CHECKPOINT_PATH, save_visuals: bool = True, verbose: bool = True):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("ZERO-DAY CAN INTRUSION DETECTION EVALUATION")
    print("=" * 70)

    if not os.path.exists(GRAPH_DATA_PATH) or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing graph data or model checkpoint in {OUTPUTS_DIR}. "
            f"Please run graph_builder.py and train.py first."
        )

    print("\n[1/4] Loading graphs, vocabulary, and model weights...")
    graphs = torch.load(GRAPH_DATA_PATH, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
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

    print(f"Loaded {len(graphs)} total graphs. Vocab size: {len(vocab)}.")
    print(f"Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test.")

    print("\n[2/4] Initializing and Calibrating Zero-Day Detector...")
    detector = ZeroDayDetector(scorer_config or AnomalyScorerConfig())
    detector.fit_baseline(train_graphs, vocab_size=len(vocab))

    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    detector.calibrate(
        model, val_loader, device=DEVICE,
        suspicious_percentile=95.0,
        alert_percentile=99.0
    )

    print("\n[3/4] Evaluating Test Set (held-out ambient + ROAD attacks + held-out synthetic episodes)...")
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    results: list[DetectionResult] = []
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
            results.append(res)

    print("\n[4/4] Computing Intrusion Detection Metrics...")
    tau = detector.tau_suspicious
    real = [r for r in results if synthetic_parent(r.capture_name) is None]
    scores = np.array([r.anomaly_score for r in real], dtype=np.float64)
    labels = np.array([r.ground_truth_label for r in real], dtype=np.int64)
    preds_binary = (scores >= tau).astype(int)
    metrics = _binary_metrics(scores, labels, tau)
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    benign_ambient = [r for r in real if str(r.capture_name).startswith("ambient_")]
    # 1 s stride -> each ambient window is one second of driving.
    metrics["false_alarm_windows_per_hour"] = (
        3600.0 * sum(r.anomaly_score >= tau for r in benign_ambient) / max(len(benign_ambient), 1))
    synthetic = summarize_synthetic(results, tau)
    syn_windows = [r for r in results if synthetic_parent(r.capture_name) is not None]
    metrics["synthetic_window_recall"] = (
        sum(r.anomaly_score >= tau for r in syn_windows) / len(syn_windows) if syn_windows else None)

    print("\n" + "=" * 70)
    print(f"REAL {DATASET.upper()} PERFORMANCE (held-out ambient + real attack captures)")
    print("=" * 70)
    print(f"  Total Test Windows      : {len(real)} (Benign: {np.sum(labels==0)}, Attack: {np.sum(labels==1)})")
    print(f"  True Positives          : {tp:5d}  |  False Positives (FP) : {fp:5d}")
    print(f"  False Negatives         : {fn:5d}  |  True Negatives (TN)  : {tn:5d}")
    print("-" * 70)
    print(f"  Decision Threshold      : {tau:.4f}")
    print(f"  High-Risk               : {detector.tau_alert:.4f}")
    print("-" * 70)
    for k in ("precision", "recall", "f1", "accuracy", "fpr", "roc_auc", "pr_auc"):
        print(f"  {k:<24}: {metrics[k]:.4f}")
    print(f"  {'false alarms / hour':<24}: {metrics['false_alarm_windows_per_hour']:.1f} (ambient driving)")
    print("=" * 70)

    if synthetic:
        print("\nSYNTHETIC TIMING-ATTACK EPISODES (reported separately; parent capture held out)")
        print(f"{'Attack_level':<22} | {'Win recall':<10} | {'Episodes':<10} | {'Median latency':<14}")
        print("-" * 66)
        for key, v in synthetic.items():
            lat = f"{v['median_latency_s']:.1f}s" if v["median_latency_s"] is not None else "N/A"
            eps = f"{v['episodes_detected']}/{v['episodes']}"
            print(f"{key:<22} | {v['window_recall']:<10.2%} | {eps:<10} | {lat:<14}")
        print(f"Overall synthetic window recall: {metrics['synthetic_window_recall']:.2%} ({len(syn_windows)} windows)")

    if verbose:
        by_capture = {}
        for r in real:
            by_capture.setdefault(r.capture_name, []).append(r)

        print("\nPER-CAPTURE ZERO-DAY EVALUATION BREAKDOWN:")
        print(f"{'Capture Name':<45} | {'Type':<10} | {'Score Mean':<10} | {'Alerts/Total':<15} | {'Attack Hits':<12} | {'Latency':<8}")
        print("-" * 110)
        for cap_name in sorted(by_capture):
            cap_results = by_capture[cap_name]
            cap_scores = [r.anomaly_score for r in cap_results]
            cap_labels = [r.ground_truth_label for r in cap_results]
            is_attack_cap = any(y == 1 for y in cap_labels)
            cap_type = "ATTACK" if is_attack_cap else "BENIGN"
            triggered_count = sum(1 for sc in cap_scores if sc >= tau)
            latency_str = "N/A"
            if is_attack_cap:
                attack_start_times = [r.window_start for r in cap_results if r.ground_truth_label == 1]
                trigger_times = [r.window_start for r in cap_results if r.anomaly_score >= tau and r.ground_truth_label == 1]
                if attack_start_times and trigger_times:
                    latency_str = f"{max(0.0, trigger_times[0] - attack_start_times[0]):.2f}s"
            n_attack = sum(cap_labels)
            attack_hits = sum(1 for r in cap_results if r.ground_truth_label == 1 and r.anomaly_score >= tau)
            hits_str = f"{attack_hits}/{n_attack}" if n_attack else "-"
            alerts = f"{triggered_count}/{len(cap_results)}"
            print(f"{cap_name:<45} | {cap_type:<10} | {np.mean(cap_scores):<10.4f} | {alerts:<15} | {hits_str:<12} | {latency_str:<8}")
        print("=" * 110)

    # Datasets with their own test subsets name captures "<subset>__<file>" (e.g. can-train-and-test
    # test_01..test_04). Report each subset on its own windows (benign + attack frames of those files).
    groups = sorted({str(r.capture_name).split("__")[0] for r in real if "__" in str(r.capture_name)})
    metrics["per_subset"] = {}
    for grp in groups:
        rs = [r for r in real if str(r.capture_name).startswith(grp + "__")]
        g_scores = np.array([r.anomaly_score for r in rs], dtype=np.float64)
        g_labels = np.array([r.ground_truth_label for r in rs], dtype=np.int64)
        metrics["per_subset"][grp] = _binary_metrics(g_scores, g_labels, tau)
        m = metrics["per_subset"][grp]
        print(f"  [{grp}] windows {len(rs)} (attack {int(g_labels.sum())}) | ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | "
              f"P {m['precision']:.3f} | R {m['recall']:.3f} | F1 {m['f1']:.3f} | FPR {m['fpr']:.3f}")

    metrics["synthetic"] = synthetic
    metrics["tau_suspicious"] = tau
    metrics["n_benign"] = int(np.sum(labels == 0))
    metrics["n_attack"] = int(np.sum(labels == 1))
    metrics["test_captures"] = list(checkpoint.get("test_captures") or [])
    if not save_visuals:
        return metrics

    cm_save_path = os.path.join(OUTPUTS_DIR, "xai_visuals", "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=labels,
        y_pred=preds_binary,
        class_names=["Benign", "Attack"],
        metrics=metrics,
        title=f"Zero-Day CAN Intrusion Detection",
        save_path=cm_save_path,
        show=False,
    )

    scenario_counts = {
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    html_save_path = os.path.join(OUTPUTS_DIR, "xai_visuals", "scenario_explainer.html")
    generate_confusion_explainer_html(counts=scenario_counts, save_path=html_save_path)

    table_save_path = os.path.join(OUTPUTS_DIR, "xai_visuals", "scenario_explainer_table.png")
    plot_scenario_explainer(counts=scenario_counts, save_path=table_save_path, show=False)

    if show_plot:
        import webbrowser
        import matplotlib.pyplot as plt
        abs_html_path = "file:///" + os.path.abspath(html_save_path).replace("\\", "/")
        print(f"\nOpening Interactive Animated Scenario Explainer in browser: {abs_html_path}")
        webbrowser.open(abs_html_path)
        print("Displaying evaluation windows (Confusion Matrix and Scenario Table) - close windows to complete.")
        plt.show()

    return metrics


def run_fold_evaluation(n_folds: int) -> dict:
    """Evaluate every capture-rotation fold; each ambient capture is tested exactly once."""
    import json
    per_fold = []
    for fold in range(n_folds):
        print(f"\n########## FOLD {fold + 1}/{n_folds} ##########")
        per_fold.append(run_evaluation(show_plot=False, checkpoint_path=fold_checkpoint_path(fold),
                                       save_visuals=False, verbose=False))
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "fpr",
            "false_alarm_windows_per_hour", "synthetic_window_recall"]
    summary = {k: {"mean": float(np.mean([m[k] for m in per_fold])), "std": float(np.std([m[k] for m in per_fold]))}
               for k in keys}
    pooled_fp = sum(m["fp"] for m in per_fold)
    pooled_benign = sum(m["n_benign"] for m in per_fold)

    print("\n" + "=" * 96)
    print(f"CAPTURE-ROTATION CROSS-VALIDATION ({n_folds} folds) - REAL ROAD METRICS")
    print("=" * 96)
    print(f"{'Fold':<5} | {'Test ambient captures':<45} | {'ROC':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'FPR':<6}")
    for i, m in enumerate(per_fold):
        caps = ",".join(c.replace("ambient_", "") for c in m["test_captures"])[:45]
        print(f"{i:<5} | {caps:<45} | {m['roc_auc']:.3f}  | {m['precision']:.3f}  | {m['recall']:.3f}  | {m['f1']:.3f}  | {m['fpr']:.3f}")
    print("-" * 96)
    for k in keys:
        print(f"  {k:<30}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")
    print(f"  {'pooled benign FPR':<30}: {pooled_fp / max(pooled_benign, 1):.4f} "
          f"({pooled_fp}/{pooled_benign} benign test windows across folds)")

    print("\nSYNTHETIC EPISODES ACROSS FOLDS (each parent capture held out exactly once)")
    for key in sorted({k for m in per_fold for k in m["synthetic"]}):
        rows = [m["synthetic"][key] for m in per_fold if key in m["synthetic"]]
        w = sum(r["windows"] for r in rows)
        hit = sum(r["window_recall"] * r["windows"] for r in rows)
        ep = sum(r["episodes"] for r in rows)
        det = sum(r["episodes_detected"] for r in rows)
        print(f"  {key:<22}: window recall {hit / max(w, 1):6.2%} ({w} windows) | episodes detected {det}/{ep}")

    out = {"per_fold": per_fold, "summary": summary, "pooled_benign_fpr": pooled_fp / max(pooled_benign, 1)}
    with open(os.path.join(OUTPUTS_DIR, "fold_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zero-Day CAN Intrusion Detection Model")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable interactive Matplotlib GUI window (saves PNG only)"
    )
    parser.add_argument("--folds", type=int, default=None, help="Evaluate all capture-rotation folds")
    args = parser.parse_args()
    if args.folds:
        run_fold_evaluation(args.folds)
    else:
        run_evaluation(show_plot=not args.no_plot)
