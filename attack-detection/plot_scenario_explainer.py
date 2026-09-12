import os
import json
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.json")

def load_index_data():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def render_card(ax, scenario_data, headers, count_info):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    bg_rect = patches.FancyBboxPatch(
        (2, 2), 96, 96,
        boxstyle="round,pad=1.5,rounding_size=3",
        facecolor="#FFFFFF",
        edgecolor="#000000",
        linewidth=1.8
    )
    ax.add_patch(bg_rect)

    header_rect = patches.FancyBboxPatch(
        (4, 83), 92, 12,
        boxstyle="round,pad=1.0,rounding_size=2",
        facecolor="#FFFFFF",
        edgecolor="#000000",
        linewidth=1.0
    )
    ax.add_patch(header_rect)

    title_text = scenario_data["name"]
    ax.text(7, 88.5, title_text, fontsize=12, fontweight="bold", color="#000000", va="center")

    if count_info:
        ax.text(7, 78.5, count_info, fontsize=9.5, fontweight="bold", color="#000000", va="center")
        y_cursor = 73.0
    else:
        y_cursor = 77.0

    ax.text(7, y_cursor, headers["vehicle_status"] + ":", fontsize=8.5, fontweight="bold", color="#000000")
    ax.text(35, y_cursor, scenario_data["vehicle_state"], fontsize=9, fontweight="bold", color="#000000")
    y_cursor -= 5.5

    ax.text(7, y_cursor, headers["model_decision"] + ":", fontsize=8.5, fontweight="bold", color="#000000")
    ax.text(35, y_cursor, scenario_data["model_verdict"], fontsize=9, fontweight="bold", color="#000000")
    y_cursor -= 8.0

    sep = patches.Rectangle((7, y_cursor + 3), 86, 0.8, facecolor="#000000", edgecolor="none")
    ax.add_patch(sep)

    ax.text(7, y_cursor, headers["system_outcome"] + ":", fontsize=8.5, fontweight="bold", color="#000000")
    y_cursor -= 4.0
    wrapped_outcome = textwrap.fill(scenario_data["real_world_consequence"], width=52)
    ax.text(7, y_cursor, wrapped_outcome, fontsize=8.2, color="#000000", va="top", linespacing=1.25)
    y_cursor -= 24.0

    ax.text(7, y_cursor, headers["model_role"] + ":", fontsize=8.5, fontweight="bold", color="#000000")
    y_cursor -= 4.0
    wrapped_role = textwrap.fill(scenario_data["role_of_model"], width=52)
    ax.text(7, y_cursor, wrapped_role, fontsize=8.2, color="#000000", va="top", linespacing=1.25)

def plot_scenario_explainer(counts=None, save_path=None, show=False):
    data = load_index_data()
    scenarios = data["scenarios"]
    plot_cfg = data["plot_strings"]
    headers = plot_cfg["section_headers"]

    fig = plt.figure(figsize=(15, 10.5), facecolor="#FFFFFF")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, hspace=0.10, wspace=0.08)

    grid_keys = [
        ("true_negative", 0, 0),
        ("false_positive", 0, 1),
        ("false_negative", 1, 0),
        ("true_positive", 1, 1)
    ]

    total_count = sum(counts.values()) if counts else 0

    for key, row, col in grid_keys:
        ax = plt.subplot2grid((2, 2), (row, col), fig=fig)
        count_str = None
        if counts and key in counts:
            c = counts[key]
            pct = (c / total_count * 100) if total_count > 0 else 0.0
            count_str = f"Window Count: {c:,} ({pct:.1f}% of evaluated traffic)"
        render_card(ax, scenarios[key], headers, count_str)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(" Saved Scenario Table to: " + save_path)

    if show:
        plt.show()

    return fig
