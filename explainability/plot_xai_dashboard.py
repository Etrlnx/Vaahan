
import os
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


def _draw_radar(ax, report, legend_border=True, legend_edgecolor="#334155", legend_linewidth=1.0):
    categories = ["Recon (R)", "Timing (T)", "Struct (G)", "Anomaly Score"]
    num_vars   = len(categories)

    r_raw = float(report.component_breakdown.get("reconstruction_error", 0.0))
    t_raw = float(report.component_breakdown.get("temporal_error",       0.0))
    g_raw = float(report.component_breakdown.get("structural_error",     0.0))
    s_raw = float(report.anomaly_score)
    tau_s = float(report.decision_threshold)
    tau_a = float(report.alert_threshold)

    norm  = tau_a if tau_a > 0 else 1.0
    cap   = norm * 1.25
    def _norm(v):
        return min(v / norm, cap / norm)

    values  = [_norm(r_raw), _norm(t_raw), _norm(g_raw), _norm(s_raw)]
    tau_s_n = tau_s / norm
    tau_a_n = 1.0

    angles  = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values  += values[:1]
    angles  += angles[:1]

    risk  = report.risk_state
    color = "#EF4444" if risk == "HIGH_RISK" else ("#F59E0B" if risk == "SUSPICIOUS" else "#10B981")

    ax.plot(angles, values, color=color, linewidth=2.5, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.30)

    circ = np.linspace(0, 2 * np.pi, 120)
    ax.plot(circ, [tau_s_n] * 120, color="#F59E0B", linestyle="--", linewidth=1.4,
            label=f"Decision  τ={tau_s:.4f}")
    ax.plot(circ, [tau_a_n] * 120, color="#EF4444", linestyle=":",  linewidth=1.7,
            label=f"Alert      τ={tau_a:.4f}")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold", color="#000000")
    ax.set_ylim(0, cap / norm)
    ax.tick_params(colors="#FFFFFF", labelsize=7.5)
    ax.grid(color="#334155", linestyle="--", alpha=0.65)
    ax.set_facecolor("#FFFFFF")

    ax.set_title(
        f"Component Deviation Profile\n",
        size=10, weight="bold", color="#000000", pad=14,
    )
    # Additional metrics: f"R={r_raw:.3f}  T={t_raw:.3f}  G={g_raw:.3f}  S={s_raw:.4f}  (normalised to τₐ={tau_a:.4f})"
    
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12),
              fontsize=8, facecolor="#FFFFFF", edgecolor=legend_edgecolor if legend_border else "none",
              linewidth=legend_linewidth, labelcolor="#000000")


def _draw_subgraph(ax, report):
    top_ids = report.top_anomalous_ids
    primary = top_ids[0]["arbitration_id"] if top_ids else "N/A"
    nodes_info = {d["arbitration_id"]: d for d in top_ids}
    edges_info = report.top_attended_interactions
    struct_edges = {f"{e['source_id']}->{e['target_id']}" for e in report.structural_anomalies}

    all_ids = list(nodes_info.keys())
    for e in edges_info:
        if e["source_id"] not in all_ids: all_ids.append(e["source_id"])
        if e["target_id"] not in all_ids: all_ids.append(e["target_id"])

    pos = {}
    ci  = all_ids.index(primary) if primary in all_ids else 0
    others = [n for i, n in enumerate(all_ids) if i != ci]
    pos[all_ids[ci]] = np.array([0.0, 0.0])
    if others:
        angs = np.linspace(0, 2 * np.pi, len(others), endpoint=False)
        for a, n in zip(angs, others):
            pos[n] = np.array([0.65 * np.cos(a), 0.65 * np.sin(a)])

    for e in edges_info:
        u, v = e["source_id"], e["target_id"]
        if u in pos and v in pos:
            rare   = f"{u}->{v}" in struct_edges
            col    = "#EF4444" if rare else "#38BDF8"
            style  = "--"      if rare else "-"
            lw     = 1.0 + float(e.get("attention_weight", 0.5)) * 3.5
            ax.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="->", color=col, lw=lw,
                                        linestyle=style, shrinkA=18, shrinkB=18, alpha=0.85))

    for nid, (x, y) in pos.items():
        is_p  = (nid == primary)
        nc    = "#EF4444" if is_p else ("#F59E0B" if nid in nodes_info else "#38BDF8")
        ns    = 1400 if is_p else 900
        ax.scatter([x], [y], s=ns * 1.3, color=nc, alpha=0.22, edgecolors="none")
        ax.scatter([x], [y], s=ns,       color=nc, alpha=0.95, edgecolors="#FFFFFF", linewidth=1.5)
        ax.text(x, y, nid, color="#FFFFFF", fontsize=9, fontweight="bold", ha="center", va="center")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    ax.set_facecolor("#1E293B")
    ax.set_title("Attended Interaction Subgraph", color="#000000", fontsize=11, fontweight="bold", pad=10)


def _draw_id_bars(ax, report):
    top_ids  = report.top_anomalous_ids
    labels   = [d["arbitration_id"] for d in top_ids][::-1]
    contribs = [float(d["anomaly_contribution_pct"]) for d in top_ids][::-1]
    cols     = ["#38BDF8"] * len(contribs)
    if cols:
        cols[-1] = "#EF4444"

    bars = ax.barh(range(len(labels)), contribs, color=cols, height=0.5, edgecolor="#ffffff")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold", color="#000000")
    ax.set_xlabel("Graph Error Share (%)", fontsize=10, fontweight="bold", color="#000000")
    ax.set_xlim(0, max(100.0, max(contribs) * 1.28 if contribs else 100.0))
    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.55)
    ax.tick_params(colors="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_title("CAN ID Anomaly Contribution", color="#000000", fontsize=11, fontweight="bold", pad=10)

    for b, v in zip(bars, contribs):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#000000")


def _draw_feature_bars(ax, report):
    top_ids = report.top_anomalous_ids
    if not top_ids:
        ax.axis("off"); return

    primary     = top_ids[0]["arbitration_id"]
    top_feats   = top_ids[0].get("top_deviating_features", [])
    if not top_feats:
        ax.axis("off"); return

    f_names = [f["feature"]  for f in top_feats][::-1]
    f_res   = [float(f["residual"]) for f in top_feats][::-1]
    f_pcts  = [float(f["feature_contribution_pct"]) for f in top_feats][::-1]
    palette = ["#10B981", "#F59E0B", "#F97316", "#EF4444", "#A855F7"]
    f_cols  = (palette * ((len(f_names) // len(palette)) + 1))[:len(f_names)]
    f_cols  = f_cols[::-1]

    bars = ax.barh(range(len(f_names)), f_res, color=f_cols, height=0.5, edgecolor="#ffffff")
    ax.set_yticks(range(len(f_names)))
    ax.set_yticklabels(f_names, fontsize=10, fontweight="bold", color="#000000")
    ax.set_xlabel(r"Reconstruction Residual $(x - \hat{x})^2 \cdot w$",
                  fontsize=10, fontweight="bold", color="#000000")
    ax.set_xlim(0, max(f_res) * 1.38 if f_res else 1.0)
    ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.55)
    ax.tick_params(colors="#94A3B8")
    ax.set_facecolor("#FFFFFF")
    ax.set_title(f"Root-Cause Features  ·  CAN ID {primary}",
                 color="#000000", fontsize=11, fontweight="bold", pad=10)

    for b, r, p in zip(bars, f_res, f_pcts):
        ax.text(b.get_width() + 0.015 * max(f_res),
                b.get_y() + b.get_height() / 2,
                f"{r:.3f}  ({p:.1f}%)",
                va="center", ha="left", fontsize=9, fontweight="bold", color="#000000")


def render_unified_dashboard(report, save_path: str = None, show: bool = False):
    if not _MPL_OK:
        print("matplotlib not available; skipping dashboard rendering.")
        return None

    fig = plt.figure(figsize=(16, 10), facecolor="#FFFFFF")

    gs = gridspec.GridSpec(2, 2,
                           hspace=0.38, wspace=0.28,
                           left=0.06, right=0.97,
                           top=0.84,  bottom=0.07)

    ax_radar   = fig.add_subplot(gs[0, 0], polar=True)
    ax_graph   = fig.add_subplot(gs[0, 1])
    ax_id_bars = fig.add_subplot(gs[1, 0])
    ax_feat    = fig.add_subplot(gs[1, 1])

    _draw_radar(ax_radar, report)
    _draw_subgraph(ax_graph, report)
    _draw_id_bars(ax_id_bars, report)
    _draw_feature_bars(ax_feat, report)

    risk = report.risk_state
    risk_color = "#EF4444" if risk == "HIGH_RISK" else ("#000000" if risk == "SUSPICIOUS" else "#000000")
    # high risk: #EF4444
    # suspicious: #F59E0B
    # low: #F59E0B
    fig.suptitle(
        f"SECURITY INCIDENT DASHBOARD  ·  [{risk}]   "
        f"Capture: {report.capture_name}   "
        f"t = {report.timestamp}s   "
        f"Score: {report.anomaly_score:.4f}",
        fontsize=12.5, fontweight="bold", color=risk_color, y=1.01,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  [XAI Visual] Saved dashboard PNG → {save_path}")

    if show:
        plt.show()

    return save_path
