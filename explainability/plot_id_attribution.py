"""
Visualization: CAN ID Anomaly Contribution & Feature Breakdown Bar Chart
-------------------------------------------------------------------------
Plots:
  1. Main Chart: Relative anomaly contribution percentages per CAN Arbitration ID.
  2. Sub-Panel / Inset: Specific feature dimension residual decomposition for
     the top-ranked anomalous CAN ID (e.g. signal_max vs mean_iat).
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None


def plot_id_attribution(report, save_path: str = None, title: str = None):
    """
    Renders a dual-panel bar chart with ID rankings and feature-level attribution.
    """
    if plt is None:
        print("matplotlib not available; skipping ID attribution plot.")
        return None

    top_ids_data = report.top_anomalous_ids
    if not top_ids_data:
        print("No anomalous ID data in report; skipping plot.")
        return None

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.3], wspace=0.35)

    # -------------------------------------------------------------
    # Panel 1: CAN Arbitration ID Anomaly Contribution (%)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    id_labels = [f"{d['arbitration_id']}\n({d.get('functional_description', '')[:16]}...)" if len(d.get('functional_description', '')) > 16 else f"{d['arbitration_id']}\n({d.get('functional_description', '')})" for d in top_ids_data]
    contrib_pcts = [float(d["anomaly_contribution_pct"]) for d in top_ids_data]

    # Invert so highest is at the top
    id_labels = id_labels[::-1]
    contrib_pcts = contrib_pcts[::-1]

    colors = ["#457B9D"] * len(contrib_pcts)
    if colors:
        colors[-1] = "#E63946"  # Highlight the #1 culprit in red

    bars1 = ax1.barh(range(len(id_labels)), contrib_pcts, color=colors, height=0.55, edgecolor="#1D3557", linewidth=1.2)
    ax1.set_yticks(range(len(id_labels)))
    ax1.set_yticklabels(id_labels, fontsize=10, fontweight="bold", color="#1D3557")
    ax1.set_xlabel("Graph Anomaly Contribution (%)", fontsize=11, fontweight="bold", color="#1D3557")
    ax1.set_xlim(0, max(100.0, max(contrib_pcts) * 1.25 if contrib_pcts else 100.0))
    ax1.set_title("Ranked CAN Arbitration IDs", fontsize=12, fontweight="bold", color="#0B090A", pad=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # Annotate bar values
    for bar, val in zip(bars1, contrib_pcts):
        ax1.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold", color="#1D3557"
        )

    # -------------------------------------------------------------
    # Panel 2: Feature Attribution Decomposition for Top-1 CAN ID
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    top_culprit = top_ids_data[0]
    top_features = top_culprit.get("top_deviating_features", [])

    if top_features:
        feat_names = [f["feature"] for f in top_features][::-1]
        feat_residuals = [float(f["residual"]) for f in top_features][::-1]
        feat_pcts = [float(f["feature_contribution_pct"]) for f in top_features][::-1]

        feat_colors = ["#2A9D8F", "#E76F51", "#F4A261", "#E9C46A"][:len(feat_names)][::-1]

        bars2 = ax2.barh(range(len(feat_names)), feat_residuals, color=feat_colors, height=0.55, edgecolor="#264653", linewidth=1.2)
        ax2.set_yticks(range(len(feat_names)))
        ax2.set_yticklabels(feat_names, fontsize=10, fontweight="bold", color="#264653")
        ax2.set_xlabel(r"Reconstruction Residual $(x - \hat{x})^2 \cdot w$", fontsize=11, fontweight="bold", color="#264653")
        ax2.set_xlim(0, max(feat_residuals) * 1.35 if feat_residuals else 1.0)
        ax2.set_title(f"Root Cause Features for {top_culprit['arbitration_id']}", fontsize=12, fontweight="bold", color="#0B090A", pad=12)
        ax2.grid(axis="x", linestyle="--", alpha=0.6)

        for bar, res, pct in zip(bars2, feat_residuals, feat_pcts):
            ax2.text(
                bar.get_width() + 0.02 * max(feat_residuals), bar.get_y() + bar.get_height() / 2,
                f"res={res:.2f} ({pct:.1f}%)", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#264653"
            )
    else:
        ax2.text(0.5, 0.5, "No feature breakdown available", ha="center", va="center")

    main_title = title or f"CAN ID Anomaly Contribution & Root-Cause Attribution\nCapture: {report.capture_name} @ t={report.timestamp}s [Score: {report.anomaly_score:.4f}]"
    plt.suptitle(main_title, fontsize=13, fontweight="bold", color="#0B090A", y=1.03)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [XAI Visual] Saved ID Attribution Chart to: {save_path}")

    plt.close(fig)
    return save_path
