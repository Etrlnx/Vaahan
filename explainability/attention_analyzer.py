
import os
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
GT_DIR = os.path.join(PARENT_DIR, "graph-transformer")
AD_DIR = os.path.join(PARENT_DIR, "attack-detection")

for p in (GT_DIR, AD_DIR, BASE_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.append(p)


@dataclass
class AttendedEdge:
    source_local_idx: int
    target_local_idx: int
    source_id_str: str
    target_id_str: str
    attention_score: float
    layer_scores: list[float]


class AttentionAnalyzer:
    def __init__(self, vocab: Optional[dict] = None):
        self.vocab = vocab or {}
        self.idx_to_id = {v: k for k, v in self.vocab.items()} if self.vocab else {}

    def set_vocab(self, vocab: dict):
        self.vocab = vocab
        self.idx_to_id = {v: k for k, v in vocab.items()}

    def get_id_str(self, vocab_idx: int) -> str:
        return self.idx_to_id.get(vocab_idx, f"ID_{vocab_idx}")

    def extract_layer_attention(self, model) -> list[tuple[torch.Tensor, torch.Tensor]]:
        raw_weights = model.get_attention_weights()
        if not raw_weights:
            return []
        return raw_weights

    def compute_edge_importance(
        self,
        batch,
        model,
        top_k: int = 5
    ) -> list[AttendedEdge]:
        raw_weights = self.extract_layer_attention(model)
        if not raw_weights or batch.edge_index.shape[1] == 0:
            return []

        num_edges = batch.edge_index.shape[1]
        num_layers = len(raw_weights)

        layer_edge_attns = []
        for l_idx, (edge_idx, weights) in enumerate(raw_weights):
            head_avg = weights.mean(dim=-1).cpu().numpy()
            layer_edge_attns.append(head_avg)

        layer_edge_attns = np.array(layer_edge_attns)
        mean_attention = layer_edge_attns.mean(axis=0)

        src_nodes = batch.edge_index[0].cpu().numpy()
        dst_nodes = batch.edge_index[1].cpu().numpy()
        global_ids = batch.id_idx.cpu().numpy()

        attended_edges = []
        for i in range(num_edges):
            u, v = src_nodes[i], dst_nodes[i]
            src_vocab_idx = int(global_ids[u])
            dst_vocab_idx = int(global_ids[v])
            src_id_str = self.get_id_str(src_vocab_idx)
            dst_id_str = self.get_id_str(dst_vocab_idx)

            attended_edges.append(AttendedEdge(
                source_local_idx=int(u),
                target_local_idx=int(v),
                source_id_str=src_id_str,
                target_id_str=dst_id_str,
                attention_score=float(mean_attention[i]),
                layer_scores=[float(layer_edge_attns[l, i]) for l in range(num_layers)],
            ))

        attended_edges.sort(key=lambda e: e.attention_score, reverse=True)
        return attended_edges[:top_k]

    def compute_attention_rollout(
        self,
        batch,
        model
    ) -> np.ndarray:
        raw_weights = self.extract_layer_attention(model)
        num_nodes = batch.x.shape[0]

        if not raw_weights or num_nodes == 0 or batch.edge_index.shape[1] == 0:
            return np.eye(num_nodes, dtype=np.float32)

        rollout = np.eye(num_nodes, dtype=np.float64)

        for edge_idx, weights in raw_weights:
            A_l = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            src = edge_idx[0].cpu().numpy()
            dst = edge_idx[1].cpu().numpy()
            head_avg = weights.mean(dim=-1).cpu().numpy()

            for s, d, w in zip(src, dst, head_avg):
                if s < num_nodes and d < num_nodes:
                    A_l[s, d] += float(w)

            A_l = 0.5 * A_l + 0.5 * np.eye(num_nodes)
            row_sums = A_l.sum(axis=1, keepdims=True)
            A_l = A_l / np.maximum(row_sums, 1e-6)

            rollout = np.matmul(A_l, rollout)

        return rollout
