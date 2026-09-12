
import os
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None


def plot_subgraph_static(report, save_path: str = None):
    if plt is None:
        return None

    top_ids = report.top_anomalous_ids
    if not top_ids:
        return None

    primary_culprit_id = top_ids[0]["arbitration_id"]
    nodes_info = {d["arbitration_id"]: d for d in top_ids}

    edges_info = report.top_attended_interactions
    structural_edges = {f"{e['source_id']}->{e['target_id']}": e for e in report.structural_anomalies}

    all_node_ids = list(nodes_info.keys())
    for e in edges_info:
        if e["source_id"] not in all_node_ids:
            all_node_ids.append(e["source_id"])
        if e["target_id"] not in all_node_ids:
            all_node_ids.append(e["target_id"])

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    n = len(all_node_ids)
    pos = {}
    center_idx = all_node_ids.index(primary_culprit_id) if primary_culprit_id in all_node_ids else 0

    other_nodes = [nid for i, nid in enumerate(all_node_ids) if i != center_idx]
    pos[all_node_ids[center_idx]] = np.array([0.0, 0.0])

    if other_nodes:
        angles = np.linspace(0, 2 * np.pi, len(other_nodes), endpoint=False)
        radius = 0.65
        for ang, nid in zip(angles, other_nodes):
            pos[nid] = np.array([radius * np.cos(ang), radius * np.sin(ang)])

    drawn_edges = set()
    for e in edges_info:
        u, v = e["source_id"], e["target_id"]
        if u in pos and v in pos:
            p1, p2 = pos[u], pos[v]
            edge_key = f"{u}->{v}"
            is_rare = edge_key in structural_edges

            edge_color = "#000000" if is_rare else "#000000"
            edge_style = "--" if is_rare else "-"
            edge_width = 1.0 + float(e.get("attention_weight", 0.5)) * 4.0

            ax.annotate(
                "", xy=p2, xytext=p1,
                arrowprops=dict(
                    arrowstyle="->", color=edge_color,
                    lw=edge_width, linestyle=edge_style,
                    shrinkA=22, shrinkB=22, alpha=0.85
                )
            )
            drawn_edges.add((u, v))

    for nid, (x, y) in pos.items():
        is_culprit = (nid == primary_culprit_id)
        node_color = "#000000" if is_culprit else ("#000000" if nid in nodes_info else "#000000")
        node_size = 1800 if is_culprit else 1200

        ax.scatter([x], [y], s=node_size * 1.35, color=node_color, alpha=0.25, edgecolors="none")
        ax.scatter([x], [y], s=node_size, color=node_color, alpha=0.95, edgecolors="#FFFFFF", linewidth=2.0)

        ax.text(x, y, nid, color="#000000", fontsize=11, fontweight="bold", ha="center", va="center")

        info = nodes_info.get(nid, {})
        desc = info.get("functional_description", "")
        if desc:
            short_desc = desc.split("/")[0].strip()
            ax.text(x, y - 0.12, short_desc, color="#000000", fontsize=8.5, ha="center", va="top", fontweight="semibold")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    title_text = f"Anomaly-Annotated Subgraph Network [{report.risk_state}]\nCapture: {report.capture_name} (t={report.timestamp}s)"
    ax.set_title(title_text, color="#000000", fontsize=13, fontweight="bold", pad=15)

    # legend_elements = [
    #     mpatches.Patch(color="#EF4444", label="Primary Anomalous CAN ID"),
    #     mpatches.Patch(color="#F59E0B", label="Contributing Anomalous ID"),
    #     mpatches.Patch(color="#38BDF8", label="Attended Neighbor ID"),
    # ]
    # ax.legend(handles=legend_elements, loc="lower right", facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  [XAI Visual] Saved Subgraph Network PNG to: {save_path}")

    plt.close(fig)
    return save_path


def export_interactive_html(report, save_path: str):
    top_ids = {d["arbitration_id"]: d for d in report.top_anomalous_ids}
    primary_id = report.top_anomalous_ids[0]["arbitration_id"] if report.top_anomalous_ids else "N/A"

    nodes_json = []
    edges_json = []

    seen_nodes = set()

    for d in report.top_anomalous_ids:
        nid = d["arbitration_id"]
        seen_nodes.add(nid)
        is_primary = (nid == primary_id)
        feat_info = "<br>".join([f"• {f['feature']}: res={f['residual']:.2f} ({f['feature_contribution_pct']}%)" for f in d.get("top_deviating_features", [])])

        nodes_json.append({
            "id": nid,
            "label": nid,
            "color": "#000000" if is_primary else "#000000",
            "size": 35 if is_primary else 24,
            "desc": d.get("functional_description", "CAN ECU Telemetry"),
            "contrib": f"{d.get('anomaly_contribution_pct', 0.0)}%",
            "features": feat_info or "Standard telemetry",
        })

    for e in report.top_attended_interactions:
        u, v = e["source_id"], e["target_id"]
        if u not in seen_nodes:
            seen_nodes.add(u)
            nodes_json.append({"id": u, "label": u, "color": "#38BDF8", "size": 18, "desc": "Neighbor ID", "contrib": "0%", "features": "N/A"})
        if v not in seen_nodes:
            seen_nodes.add(v)
            nodes_json.append({"id": v, "label": v, "color": "#38BDF8", "size": 18, "desc": "Neighbor ID", "contrib": "0%", "features": "N/A"})

        edges_json.append({
            "from": u, "to": v,
            "value": float(e.get("attention_weight", 0.5)),
            "title": f"Attention Weight: {e.get('attention_weight', 0.0):.4f}"
        })

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>XAI Subgraph - {report.incident_id}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; background:
    .card {{ background:
    .card h3 {{ margin-top: 0; font-size: 14px; color:
    .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }}
    .metric-val {{ font-weight: bold; color:
  </style>
</head>
<body>
  <div id="header">
    <h2>Interactive Anomaly Subgraph — Incident {report.incident_id}</h2>
    <div><span class="badge">{report.risk_state} (Score: {report.anomaly_score:.4f})</span></div>
  </div>
  <div id="container">
    <div id="mynetwork"></div>
    <div id="sidebar">
      <div class="card">
        <h3>INCIDENT TELEMETRY</h3>
        <div class="metric-row"><span>Capture:</span><span class="metric-val">{report.capture_name}</span></div>
        <div class="metric-row"><span>Timestamp:</span><span class="metric-val">{report.timestamp}s</span></div>
        <div class="metric-row"><span>Recon Residual (R):</span><span class="metric-val">{report.component_breakdown.get('reconstruction_error', 0.0):.4f}</span></div>
        <div class="metric-row"><span>Temporal Jitter (T):</span><span class="metric-val">{report.component_breakdown.get('temporal_error', 0.0):.4f}</span></div>
        <div class="metric-row"><span>Structural Penalty (G):</span><span class="metric-val">{report.component_breakdown.get('structural_error', 0.0):.4f}</span></div>
      </div>
      <div class="card">
        <h3>GATEWAY ENFORCEMENT</h3>
        <div style="font-size: 13px; color: #FCA5A5; line-height: 1.4;">{report.recommended_gateway_action}</div>
      </div>
      <div id="nodeDetails" class="card" style="display:none;">
        <h3 id="detailId">NODE DETAILS</h3>
        <div id="detailContent" style="font-size: 12.5px; line-height: 1.5; color: #CBD5E1;"></div>
      </div>
    </div>
  </div>

  <script>
    const nodes = new vis.DataSet({json.dumps(nodes_json)});
    const edges = new vis.DataSet({json.dumps(edges_json)});

    const container = document.getElementById('mynetwork');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        font: {{ color: '#FFFFFF', size: 14, face: 'sans-serif' }},
        borderWidth: 2,
        shadow: true
      }},
      edges: {{
        arrows: 'to',
        color: {{ color: '#38BDF8', highlight: '#EF4444', hover: '#F59E0B' }},
        smooth: {{ type: 'continuous' }}
      }},
      physics: {{
        barnesHut: {{ gravitationalConstant: -4000, centralGravity: 0.3, springLength: 120 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 100 }}
    }};

    const network = new vis.Network(container, data, options);

    network.on("selectNode", function (params) {{
      const nodeId = params.nodes[0];
      const node = nodes.get(nodeId);
      if (node) {{
        document.getElementById('nodeDetails').style.display = 'block';
        document.getElementById('detailId').innerText = 'CAN ID: ' + node.id;
        document.getElementById('detailContent').innerHTML = 
          '<b>Subsystem:</b> ' + node.desc + '<br>' +
          '<b>Graph Contribution:</b> ' + node.contrib + '<br><br>' +
          '<b>Top Deviating Features:</b><br>' + node.features;
      }}
    }});
  </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [XAI Visual] Saved Interactive Subgraph HTML to: {save_path}")
    return save_path


def plot_interactive_subgraph(report, png_save_path: str = None, html_save_path: str = None):
    static_res = plot_subgraph_static(report, save_path=png_save_path) if png_save_path else None
    html_res = export_interactive_html(report, save_path=html_save_path) if html_save_path else None
    return static_res, html_res
