
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Benign validation 99th percentiles of each raw component (radar spoke = value / p99).
TAU_R = 0.5456
TAU_T = 0.1649
TAU_G = 2.8964
TAU_E = 838.29


def plot_component_radar(report, save_path: str = None, title: str = None):
    if plt is None:
        print("matplotlib not available; skipping radar plot.")
        return None

    categories = [
        "Reconstruction (R)",
        "Temporal Jitter (T)",
        "Structural NLL (G)",
        "Transition Excess (E)",
        "Fused Anomaly Score",
    ]
    num_vars = len(categories)

    r_val = float(report.component_breakdown.get("reconstruction_error", 0.0))
    t_val = float(report.component_breakdown.get("temporal_error", 0.0))
    g_val = float(report.component_breakdown.get("structural_error", 0.0))
    e_val = float(report.component_breakdown.get("excess_error", 0.0))
    score_val = float(report.anomaly_score)

    tau_suspicious = float(report.decision_threshold)
    tau_alert = float(report.alert_threshold)

    r_norm = r_val / max(TAU_R, 1e-4)
    t_norm = t_val / max(TAU_T, 1e-4)
    g_norm = g_val / max(TAU_G, 1e-4)
    e_norm = max(e_val, 0.0) / max(TAU_E, 1e-4)
    s_norm = score_val / max(tau_suspicious, 1e-4)

    values = [r_norm, t_norm, g_norm, e_norm, s_norm]
    alert_ratio = tau_alert / max(tau_suspicious, 1e-4)

    max_val = max(1.8, max(values) * 1.15, alert_ratio * 1.1)

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

    ax.plot(angles, values, color=line_color, linewidth=2.5, linestyle="solid", label="Observed Deviation Profile")
    ax.fill(angles, values, color=fill_color, alpha=0.35)

    threshold_angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        threshold_angles, [1.0] * 100,
        color="#F77F00", linestyle="--", linewidth=1.5,
        label="Decision Threshold (1.0x \u03c4)"
    )
    ax.plot(
        threshold_angles, [alert_ratio] * 100,
        color="#D62828", linestyle=":", linewidth=1.8,
        label=f"Alert Threshold ({alert_ratio:.2f}x \u03c4)"
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold", color="#1D3557")

    ax.set_ylim(0, max_val)
    ax.set_rlabel_position(45)
    ticks = [0.5, 1.0, 1.5, round(max_val, 1)]
    tick_labels = ["0.5x", "1.0x (\u03c4)", "1.5x", f"{round(max_val, 1):.1f}x"]
    plt.yticks(ticks, tick_labels, color="#6C757D", size=9)

    chart_title = title or f"Normalized Deviation Profile [{report.risk_state}]\nCapture: {report.capture_name} \n @ t={report.timestamp}s"
    plt.title(chart_title, size=13, weight="bold", color="#0B090A", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [XAI Visual] Saved Radar Chart to: {save_path}")

    plt.close(fig)
    return save_path
