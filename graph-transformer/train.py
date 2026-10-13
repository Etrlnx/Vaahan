import os
import sys
import copy
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GRAPH_DATA_PATH = os.path.join(OUTPUT_DIR, "graphs.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
VAL_FRACTION = 0.15
CALIB_TAIL = 0.0        # >0 moves the last fraction of each train capture into validation (tried 0.2: lower fold means)
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import GraphTransformerAutoencoder, ModelConfig, reconstruction_loss
from preprocess import WINDOW_SIZE_SEC


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SYNTHETIC_PREFIX = "synthetic__"


def synthetic_parent(capture_name: str):
    """`synthetic__<parent>__<attack>` -> `<parent>`; None for real captures."""
    name = str(capture_name)
    return name.split("__")[1] if name.startswith(SYNTHETIC_PREFIX) else None


def split_captures(ambient_captures: list, val_fraction: float = VAL_FRACTION, seed: int = SEED,
                   fold: int = None, n_folds: int = None):
    """Capture-level split. Default: one seeded split (15% val, 15% of rest test).
    Fold mode: shuffled captures are cut into n_folds groups; group `fold` is test, the
    first capture of the next group is val, the rest train - every capture is tested once."""
    captures = sorted(ambient_captures)
    random.Random(seed).shuffle(captures)
    if fold is None:
        n_val = max(1, int(len(captures) * val_fraction))
        remaining = captures[n_val:]
        n_test = max(1, int(len(remaining) * 0.15))
        return remaining[n_test:], captures[:n_val], remaining[:n_test]
    groups = [list(g) for g in np.array_split(captures, n_folds)]
    test = groups[fold]
    val = groups[(fold + 1) % n_folds][:1]
    train = [c for c in captures if c not in test and c not in val]
    return train, val, test


def is_test_graph(g, test_ambient_captures) -> bool:
    """Test = held-out ambient captures, every real attack-folder capture, and only the
    attack-labelled windows of synthetic captures whose parent capture is held out."""
    name = str(g.capture_name)
    parent = synthetic_parent(name)
    if parent is not None:
        return parent in test_ambient_captures and g.y.item() == 1
    return name in test_ambient_captures or not name.startswith("ambient_") or g.y.item() == 1


def partition_benign(graphs: list, train_captures, val_captures, calib_tail: float = CALIB_TAIL):
    """Benign train/val windows. Val = whole val captures + the time-ordered tail of every train
    capture, so thresholds see every driving domain (dyno and highway), not one drive. Windows
    straddling the tail cutoff are dropped so no CAN message is in both train and val."""
    benign = [g for g in graphs if g.y.item() == 0]
    val = [g for g in benign if g.capture_name in val_captures]
    train = []
    if calib_tail <= 0:
        return [g for g in benign if g.capture_name in train_captures], val
    last_start = {}
    for g in benign:
        if g.capture_name in train_captures:
            last_start[g.capture_name] = max(last_start.get(g.capture_name, 0.0), float(g.window_start))
    for g in benign:
        if g.capture_name not in train_captures:
            continue
        cutoff = (1.0 - calib_tail) * (last_start[g.capture_name] + WINDOW_SIZE_SEC)
        start = float(g.window_start)
        if start + WINDOW_SIZE_SEC <= cutoff:
            train.append(g)
        elif start >= cutoff:
            val.append(g)
    return train, val


def split_data(graphs: list, val_fraction: float = VAL_FRACTION, seed: int = SEED,
               fold: int = None, n_folds: int = None):
    ambient_names = {g.capture_name for g in graphs if str(g.capture_name).startswith("ambient_")}
    train_captures, val_captures, test_ambient_captures = split_captures(
        list(ambient_names), val_fraction, seed, fold, n_folds)

    train_benign, val_benign = partition_benign(graphs, train_captures, val_captures)
    test_set = [g for g in graphs if is_test_graph(g, test_ambient_captures)]

    attack_captures = sorted({g.capture_name for g in test_set if not str(g.capture_name).startswith("ambient_")})

    print(f"Split captures: {len(train_captures)} train ambient, {len(val_captures)} val ambient, "
          f"{len(test_ambient_captures)} test ambient, {len(attack_captures)} attack captures.")
    print(f"Split graphs: {len(train_benign)} train (ambient), {len(val_benign)} val (ambient), "
          f"{len(test_set)} test total.")

    return train_benign, val_benign, test_set, train_captures, val_captures, test_ambient_captures


def align_graph_feature_dim(graphs, expected_dim: int):
    aligned = 0
    for g in graphs:
        if g.x.size(-1) != expected_dim:
            if g.x.size(-1) < expected_dim:
                pad = torch.zeros((g.x.size(0), expected_dim - g.x.size(-1)), dtype=g.x.dtype)
                g.x = torch.cat([g.x, pad], dim=1)
            else:
                g.x = g.x[:, :expected_dim]
            aligned += 1
    if aligned:
        print(f"Aligned {aligned} graphs to feature dimension {expected_dim}.")


def normalize_graph_features(train_graphs, val_graphs, test_graphs):  
    if not train_graphs:
        raise ValueError("Training split is empty; cannot compute feature normalization stats.")

    train_x = torch.cat([g.x for g in train_graphs], dim=0)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    for graph_list in (train_graphs, val_graphs, test_graphs):
        for g in graph_list:
            g.x = ((g.x - mean) / std).to(torch.float32)

    return mean, std


def apply_saved_normalization(graphs: list, mean: torch.Tensor, std: torch.Tensor):
    mean = mean.to(torch.float32)
    std = std.to(torch.float32)
    for g in graphs:
        g.x = ((g.x - mean) / std).to(torch.float32)


def run_epoch(model, loader, optimizer, config, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_graphs = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            loss = reconstruction_loss(outputs, batch.x, batch.edge_index, config, model=model)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

    return total_loss / max(n_graphs, 1)


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


def compute_per_graph_reconstruction_error(model, loader, config):
    model.eval()
    results = []
    feature_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5],
        device=DEVICE,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if batch.num_graphs != 1:
                raise ValueError("Evaluation loader must use batch_size=1 to retain per-capture metadata.")

            outputs = model(
                x_stats=batch.x,
                id_idx=batch.id_idx,
                edge_index=batch.edge_index,
                edge_weight=batch.edge_attr,
            )
            node_se = (((outputs["x_recon"] - batch.x) ** 2) * feature_weights).mean(dim=-1)
            node_error = node_se.mean().item()

            if config.reconstruct_edges and outputs["edge_logits"] is not None and batch.edge_index.shape[1] > 0:
                pos_labels = torch.ones_like(outputs["edge_logits"])
                edge_loss = F.binary_cross_entropy_with_logits(outputs["edge_logits"], pos_labels).item()
                graph_error = node_error + 0.5 * edge_loss
            else:
                graph_error = node_error

            raw_label = batch.y[0].item()
            raw_capture = batch.capture_name
            if isinstance(raw_capture, (list, tuple)):
                capture_name = raw_capture[0]
            else:
                capture_name = raw_capture

            results.append((graph_error, raw_label, capture_name))

    return results


FOLDS_DIR = os.path.join(OUTPUT_DIR, "folds")


def fold_checkpoint_path(fold: int) -> str:
    return os.path.join(FOLDS_DIR, f"fold_{fold}.pt")


def main(fold: int = None, n_folds: int = None):
    set_seed(SEED)
    checkpoint_path = CHECKPOINT_PATH if fold is None else fold_checkpoint_path(fold)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print("Loading graphs and vocab...")
    graphs = torch.load(GRAPH_DATA_PATH, weights_only=False)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs, vocab size {len(vocab)}")

    config = ModelConfig()
    config.num_ids = len(vocab)

    align_graph_feature_dim(graphs, config.node_stat_feature_dim)
    train_graphs, val_graphs, test_graphs, train_caps, val_caps, test_caps = split_data(graphs, fold=fold, n_folds=n_folds)
    train_mean, train_std = normalize_graph_features(train_graphs, val_graphs, test_graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    print(f"Train-set feature stats: mean={train_mean[:4].tolist()} | std={train_std[:4].tolist()}")

    model = GraphTransformerAutoencoder(config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {DEVICE}...")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, config, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, config, train=False)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
                "train_mean": train_mean.cpu(),
                "train_std": train_std.cpu(),
                "train_captures": train_caps,
                "val_captures": val_caps,
                "calib_tail": CALIB_TAIL,
                "test_captures": test_caps,
                "seed": SEED,
                "fold": fold,
                "n_folds": n_folds,
                "vocab": vocab,
            }, checkpoint_path)

    print("\nEvaluating on test set (held-out benign + all attack windows)...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    graph_results = compute_per_graph_reconstruction_error(model, test_loader, config)
    errors = np.array([result[0] for result in graph_results], dtype=np.float32)
    labels = np.array([result[1] for result in graph_results], dtype=np.int64)

    per_capture_errors = {}
    for err, label, capture_name in graph_results:
        per_capture_errors.setdefault(capture_name, []).append((err, label))

    print("\nPer-capture reconstruction breakdown:")
    for capture_name in sorted(per_capture_errors):
        values = np.array([entry[0] for entry in per_capture_errors[capture_name]], dtype=np.float32)
        n_benign = sum(1 for entry in per_capture_errors[capture_name] if entry[1] == 0)
        n_attack = sum(1 for entry in per_capture_errors[capture_name] if entry[1] == 1)
        print(f"  {capture_name:45s}: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"n_benign={n_benign}, n_attack={n_attack}")

    benign_errors = errors[labels == 0]
    attack_errors = errors[labels == 1]
    roc_auc = compute_roc_auc(errors, labels)

    print(f"\nAggregate Evaluation Summary:")
    print(f"  Benign reconstruction error : mean={benign_errors.mean():.4f}, std={benign_errors.std():.4f}, n={len(benign_errors)}")
    print(f"  Attack reconstruction error : mean={attack_errors.mean():.4f}, std={attack_errors.std():.4f}, n={len(attack_errors)}")
    print(f"  Reconstruction ROC-AUC      : {roc_auc:.4f}")

    if attack_errors.mean() > benign_errors.mean():
        print("\n-> Attack windows show HIGHER reconstruction error on average.")
    else:
        print("\n-> Attack windows do NOT show higher reconstruction error on average.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the graph transformer autoencoder")
    parser.add_argument("--fold", type=int, default=None, help="Fold index for capture-rotation CV")
    parser.add_argument("--n-folds", type=int, default=None, help="Number of capture-rotation folds")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (for parallel folds)")
    args = parser.parse_args()
    if (args.fold is None) != (args.n_folds is None):
        parser.error("--fold and --n-folds must be given together")
    if args.threads:
        torch.set_num_threads(args.threads)
    main(args.fold, args.n_folds)
