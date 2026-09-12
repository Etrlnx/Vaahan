
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_component_radar(report, save_path: str = None, title: str = None):
    if plt is None:
        print("matplotlib not available; skipping radar plot.")
        return None

    categories = [
        "Reconstruction (R)",
        "Temporal Jitter (T)",
        "Structural NLL (G)",
        "Fused Anomaly Score",
    ]
    num_vars = len(categories)

    r_val = float(report.component_breakdown.get("reconstruction_error", 0.0))
    t_val = float(report.component_breakdown.get("temporal_error", 0.0))
    g_val = float(report.component_breakdown.get("structural_error", 0.0))
    score_val = float(report.anomaly_score)

    tau_suspicious = float(report.decision_threshold)
    tau_alert = float(report.alert_threshold)

    max_val = max(1.5, r_val, t_val, g_val, score_val, tau_alert) * 1.15
    values = [r_val, t_val, g_val, score_val]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    if report.risk_state == "HIGH_RISK":
        line_color = "#D90429"
        fill_color = "#EF233C"
    elif report.risk_state == "SUSPICIOUS":
        line_color = "#FF9F1C"
        fill_color = "#FFE49E"
    else:
        line_color = "#2EC4B6"
        fill_color = "#CBF3F0"

    ax.plot(angles, values, color=line_color, linewidth=2.5, linestyle="solid", label="Observed Anomaly Profile")
    ax.fill(angles, values, color=fill_color, alpha=0.35)

    threshold_angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        threshold_angles, [tau_suspicious] * 100,
        color="#F77F00", linestyle="--", linewidth=1.5,
        label=f"Decision Threshold (τ={tau_suspicious:.2f})"
    )
    ax.plot(
        threshold_angles, [tau_alert] * 100,
        color="#D62828", linestyle=":", linewidth=1.8,
        label=f"Alert Threshold (τ={tau_alert:.2f})"
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold", color="#1D3557")

    ax.set_ylim(0, max_val)
    ax.set_rlabel_position(45)
    plt.yticks(
        np.linspace(0.2, round(max_val, 1), 4),
        [f"{v:.1f}" for v in np.linspace(0.2, round(max_val, 1), 4)],
        color="#6C757D", size=9
    )

    chart_title = title or f"Component Deviation Profile [{report.risk_state}]\nCapture: {report.capture_name} \n @ t={report.timestamp}s"
    plt.title(chart_title, size=13, weight="bold", color="#0B090A", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [XAI Visual] Saved Radar Chart to: {save_path}")

    plt.close(fig)
    return save_path
