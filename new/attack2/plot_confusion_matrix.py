import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    metrics: dict = None,
    title: str = "CAN Intrusion Detection",
    save_path: str = "./attack-detection/confmat",
    show: bool = False,
):
    if not _MPL_OK:
        print("[Warning] Matplotlib not available; skipping confusion matrix plot.")
        return None

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if class_names is None:
        class_names = ["Benign", "Zero-Day"]

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    total = np.sum(cm)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / max(total, 1)
    fpr = fp / max(fp + tn, 1)
    specificity = tn / max(tn + fp, 1)

    roc_auc = metrics.get("roc_auc") if metrics else None
    pr_auc = metrics.get("pr_auc") if metrics else None

    fig = plt.figure(figsize=(10, 8), facecolor="#FFFFFF")
    ax = fig.add_subplot(111, facecolor="#FFFFFF")

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greys, vmin=0.0, vmax=2.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#000000", labelsize=9)
    cbar.set_label("Row Normalized Rate", color="#000000", fontsize=10, weight="bold")
    # Row Normalized Rate = Recall / Specificity
    cbar.outline.set_edgecolor("#000000")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold", color="#000000")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="bold", color="#000000")
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", color="#000000", labelpad=10)
    ax.set_ylabel("Ground-Truth", fontsize=12, fontweight="bold", color="#000000", labelpad=10)
    ax.tick_params(colors="#000000", length=0)

    cell_values = [
        [tn, fp],
        [fn, tp],
    ]

    for i in range(2):
        for j in range(2):
            count = cell_values[i][j]
            ax.text(j, i, f"{count:,}", ha="center", va="center", color="#000000", fontsize=18, fontweight="bold")

    ax.set_xticks(np.arange(2) - 0.5, minor=True)
    ax.set_yticks(np.arange(2) - 0.5, minor=True)
    ax.grid(which="minor", color="#000000", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_edgecolor("#000000")
        spine.set_linewidth(1.5)

    ax.set_title(title, fontsize=13, fontweight="bold", color="#000000", pad=15)

    summary_text = (
        f"Accuracy: {accuracy:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  "
        f"F1-Score: {f1:.4f}  |  FPR: {fpr:.4f}"
    )
    if roc_auc is not None and pr_auc is not None:
        summary_text += f"\nROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}  |  Total Windows: {total:,}"

    fig.text(
        0.5, 0.02, summary_text,
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
        color="#000000",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFFF", edgecolor="#000000", linewidth=1.5)
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  [Evaluation Visual] Saved Confusion Matrix to: {save_path}")

    if show:
        plt.show()

    return fig
