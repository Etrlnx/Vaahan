import pickle
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from preprocess import WindowRecord, discover_captures, load_capture, window_capture, AMBIENT_DIR, ATTACK_DIR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from preprocess import OUTPUT_DIR
GRAPH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "graphs.pt")
VOCAB_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vocab.pkl")
NODE_FEATURE_DIM = 9

def build_id_vocab(windows: list[WindowRecord]) -> dict:
    """Vocabulary from benign ambient captures only; IDs seen only in attack or synthetic
    captures map to <UNK> at inference instead of getting an untrained embedding."""
    all_ids = set()
    for idx, w in enumerate(windows):
        if not str(w.capture_name).startswith("ambient_"):
            continue
        if w.messages is None:
            raise ValueError(
                f"Window {idx} ({w.capture_name}) has no raw messages. "
            ) 
        all_ids.update(w.messages["ID"].unique().tolist())

    vocab = {"<UNK>": 0}
    for i, id_str in enumerate(sorted(all_ids), start=1):
        vocab[id_str] = i

    return vocab


def _extract_signal_columns(df: pd.DataFrame) -> list[str]:
    
    return [c for c in df.columns if c.startswith("Signal_")]


def extract_node_features(id_messages: pd.DataFrame, window_duration: float, total_window_messages: int) -> np.ndarray:
    signal_cols = _extract_signal_columns(id_messages)

    msg_count = len(id_messages)
    activity_share = float(msg_count / max(total_window_messages, 1))

    times = id_messages["Time"].values
    if len(times) > 1:
        iats = np.diff(times)
        mean_iat = float(np.mean(iats))
        std_iat = float(np.std(iats))
    else:
        mean_iat = window_duration
        std_iat = 0.0

    if signal_cols:
        raw_signals = id_messages[signal_cols].values.flatten()
        raw_signals = raw_signals[~np.isnan(raw_signals)]
    else:
        raw_signals = np.array([])

    if len(raw_signals) > 0:
        signal_values = np.sign(raw_signals) * np.log1p(np.abs(raw_signals))
        signal_mean = float(np.mean(signal_values))
        signal_std = float(np.std(signal_values))
        signal_min = float(np.min(signal_values))
        signal_max = float(np.max(signal_values))
    else:
        signal_mean = signal_std = signal_min = signal_max = 0.0

    signal_range = float(signal_max - signal_min)
    n_signal_channels = float(len(signal_cols))

    return np.array([
        msg_count,
        mean_iat,
        std_iat,
        signal_mean,
        signal_std,
        signal_min,
        signal_max,
        activity_share,
        signal_range
    ], dtype=np.float32)


def build_edges(messages: pd.DataFrame, node_index: dict) -> tuple[np.ndarray, np.ndarray]:
    """Directed transition counts between consecutive messages of different IDs, edges in order
    of first occurrence (same output as the original per-message loop, vectorised)."""
    ids_in_order = messages.sort_values("Time")["ID"].values
    codes = pd.Index(list(node_index)).get_indexer(ids_in_order)
    src, dst = codes[:-1], codes[1:]
    keep = src != dst
    if not keep.any():
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    n = len(node_index)
    pairs = src[keep].astype(np.int64) * n + dst[keep]
    uniq, first, counts = np.unique(pairs, return_index=True, return_counts=True)
    order = np.argsort(first)
    edges = np.stack([uniq[order] // n, uniq[order] % n])
    return edges, counts[order].astype(np.float32)


def build_graph_for_window(window: WindowRecord, global_vocab: dict) -> Data:
    df = window.messages
    unique_ids = df["ID"].unique().tolist()

    local_node_index = {id_str: i for i, id_str in enumerate(unique_ids)}

    window_duration = window.window_end - window.window_start
    total_window_messages = len(df)

    node_features = []
    global_id_indices = []

    groups = dict(tuple(df.groupby("ID", sort=False)))
    for id_str in unique_ids:
        id_msgs = groups[id_str]
        feat = extract_node_features(id_msgs, window_duration, total_window_messages)
        node_features.append(feat)
        global_id_indices.append(global_vocab.get(id_str, global_vocab["<UNK>"]))

    x = torch.tensor(np.stack(node_features), dtype=torch.float32)
    id_idx = torch.tensor(global_id_indices, dtype=torch.long)

    edge_index_np, edge_weight_np = build_edges(df, local_node_index)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor([window.label], dtype=torch.long),
    )
    data.id_idx = id_idx
    data.capture_name = window.capture_name
    data.window_start = window.window_start
    data.frac_attack_messages = window.frac_attack_messages
    data.graph_message_count = float(len(df))
    data.graph_unique_ids = float(len(unique_ids))
    data.graph_attack_fraction = float(df["Label"].mean())

    return data


def _capture_windows(capture):
    df = load_capture(capture)
    interval = capture.metadata.get("injection_interval") if capture.metadata.get("synthetic") else None
    return window_capture(df, capture.capture_name, capture.is_attack, keep_messages=True, attack_interval=interval)


def main():
    """Windows and builds graphs one capture at a time, so peak memory is one capture's raw
    messages rather than every window of the dataset (the old pickle route ran out of RAM)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    captures = discover_captures(AMBIENT_DIR, ATTACK_DIR)

    print("Building global ID vocabulary from ambient captures...")
    ids = set()
    for c in captures:
        if c.capture_name.startswith("ambient_"):
            ids.update(k for k in build_id_vocab(_capture_windows(c)) if k != "<UNK>")
    vocab = {"<UNK>": 0, **{id_str: i for i, id_str in enumerate(sorted(ids), start=1)}}
    print(f"  -> {len(vocab)} unique IDs (including <UNK>)")

    print("Building graphs capture by capture...")
    graphs = []
    skipped = 0
    for n, capture in enumerate(captures, 1):
        windows = _capture_windows(capture)
        for w in windows:
            try:
                g = build_graph_for_window(w, vocab)
                if g.x.shape[0] < 2:
                    skipped += 1
                    continue
                graphs.append(g)
            except Exception as e:
                print(f"  WARNING: failed to build graph for {w.capture_name} @ {w.window_start}s: {e}")
                skipped += 1
        print(f"  [{n}/{len(captures)}] {capture.capture_name}: {len(windows)} windows")

    print(f"Built {len(graphs)} graphs, skipped {skipped}.")

    torch.save(graphs, GRAPH_OUTPUT_PATH)
    with open(VOCAB_OUTPUT_PATH, "wb") as f:
        pickle.dump(vocab, f)

    import json
    meta_path = os.path.join(OUTPUT_DIR, "graphs_metadata.json")
    meta = {
        "node_feature_dim": NODE_FEATURE_DIM,
        "vocab_size": len(vocab),
        "total_graphs": len(graphs),
        "skipped_graphs": skipped,
        "benign_graphs": sum(g.y.item() == 0 for g in graphs),
        "attack_graphs": sum(g.y.item() == 1 for g in graphs),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graphs to {GRAPH_OUTPUT_PATH}")
    print(f"Saved vocab to {VOCAB_OUTPUT_PATH}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
