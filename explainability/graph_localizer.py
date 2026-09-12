
import os
import sys
from dataclasses import dataclass, field
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

from scorer import TransitionBaseline
from attention_analyzer import AttentionAnalyzer


@dataclass
class LocalEdgeDetail:
    source_id: str
    target_id: str
    source_vocab_idx: int
    target_vocab_idx: int
    transition_count: float
    attention_weight: float
    transition_nll: float
    is_rare_transition: bool


@dataclass
class LocalSubgraph:
    center_id: str
    center_vocab_idx: int
    neighbor_ids: list[str]
    edges: list[LocalEdgeDetail]
    subgraph_attention_mass: float
    max_transition_penalty: float


class GraphLocalizer:
    def __init__(
        self,
        vocab: Optional[dict] = None,
        transition_baseline: Optional[TransitionBaseline] = None,
        rare_prob_threshold: float = 1e-4
    ):
        self.vocab = vocab or {}
        self.idx_to_id = {v: k for k, v in self.vocab.items()} if self.vocab else {}
        self.transition_baseline = transition_baseline
        self.rare_prob_threshold = rare_prob_threshold

    def set_vocab(self, vocab: dict):
        self.vocab = vocab
        self.idx_to_id = {v: k for k, v in vocab.items()}

    def get_id_str(self, vocab_idx: int) -> str:
        return self.idx_to_id.get(vocab_idx, f"ID_{vocab_idx}")

    def localize_subgraph(
        self,
        batch,
        model,
        center_local_idx: int,
        attention_analyzer: Optional[AttentionAnalyzer] = None,
    ) -> LocalSubgraph:
        num_nodes = batch.x.shape[0]
        global_ids = batch.id_idx.cpu().numpy()
        center_vocab_idx = int(global_ids[center_local_idx])
        center_id_str = self.get_id_str(center_vocab_idx)

        if batch.edge_index.shape[1] == 0:
            return LocalSubgraph(
                center_id=center_id_str,
                center_vocab_idx=center_vocab_idx,
                neighbor_ids=[],
                edges=[],
                subgraph_attention_mass=0.0,
                max_transition_penalty=0.0,
            )

        src_nodes = batch.edge_index[0].cpu().numpy()
        dst_nodes = batch.edge_index[1].cpu().numpy()
        weights = batch.edge_attr.cpu().numpy() if hasattr(batch, "edge_attr") and batch.edge_attr is not None else np.ones(len(src_nodes))

        edge_attentions = {}
        if attention_analyzer is not None:
            raw_weights = attention_analyzer.extract_layer_attention(model)
            if raw_weights:
                layer_attns = [w.mean(dim=-1).cpu().numpy() for _, w in raw_weights]
                mean_attn = np.mean(layer_attns, axis=0)
                for e_idx in range(len(mean_attn)):
                    edge_attentions[e_idx] = float(mean_attn[e_idx])

        connected_edges = []
        neighbor_local_indices = set()
        subgraph_attention = 0.0
        max_penalty = 0.0

        for e_idx, (u, v, count) in enumerate(zip(src_nodes, dst_nodes, weights)):
            if u == center_local_idx or v == center_local_idx:
                other_idx = v if u == center_local_idx else u
                neighbor_local_indices.add(int(other_idx))

                u_vocab = int(global_ids[u])
                v_vocab = int(global_ids[v])
                u_str = self.get_id_str(u_vocab)
                v_str = self.get_id_str(v_vocab)

                attn_val = edge_attentions.get(e_idx, 0.0)
                subgraph_attention += attn_val

                nll_val = 0.0
                is_rare = False
                if self.transition_baseline is not None and self.transition_baseline.fitted:
                    if u_vocab < self.transition_baseline.vocab_size and v_vocab < self.transition_baseline.vocab_size:
                        prob = float(self.transition_baseline.transition_probs[u_vocab, v_vocab])
                        nll_val = -float(np.log(max(prob, 1e-7)))
                        is_rare = prob < self.rare_prob_threshold
                        max_penalty = max(max_penalty, nll_val)

                connected_edges.append(LocalEdgeDetail(
                    source_id=u_str,
                    target_id=v_str,
                    source_vocab_idx=u_vocab,
                    target_vocab_idx=v_vocab,
                    transition_count=float(count),
                    attention_weight=attn_val,
                    transition_nll=nll_val,
                    is_rare_transition=is_rare,
                ))

        neighbor_strs = [self.get_id_str(int(global_ids[n])) for n in neighbor_local_indices]

        return LocalSubgraph(
            center_id=center_id_str,
            center_vocab_idx=center_vocab_idx,
            neighbor_ids=neighbor_strs,
            edges=connected_edges,
            subgraph_attention_mass=subgraph_attention,
            max_transition_penalty=max_penalty,
        )
