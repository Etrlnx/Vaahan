import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import negative_sampling


class ModelConfig:
    num_ids = 106

    node_stat_feature_dim = 9

    id_embedding_dim = 20
    hidden_dim = 96
    latent_dim = 48
    num_transformer_layers = 4
    num_attention_heads = 4
    dropout = 0.08

    reconstruct_edges = True
    neg_sample_ratio: float = 1.0


class GraphTransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        assert config.num_ids is not None, (
            "config.num_ids must be set to len(id_vocab) before constructing the model. "
            "See train.py for how this is wired up from graph_builder.py's saved vocab."
        )

        self.id_embedding = nn.Embedding(config.num_ids, config.id_embedding_dim)

        input_dim = config.node_stat_feature_dim + config.id_embedding_dim
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        self.transformer_layers = nn.ModuleList()
        in_dim = config.hidden_dim
        for i in range(config.num_transformer_layers):
            out_dim = config.hidden_dim if i < config.num_transformer_layers - 1 else config.latent_dim
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=in_dim,
                    out_channels=out_dim // config.num_attention_heads,
                    heads=config.num_attention_heads,
                    dropout=config.dropout,
                    edge_dim=1,
                    concat=True,
                )
            )
            in_dim = out_dim

        self.dropout = nn.Dropout(config.dropout)

        self.last_attention_weights = []

    def forward(self, x_stats, id_idx, edge_index, edge_weight):

        id_emb = self.id_embedding(id_idx)
        h = torch.cat([x_stats, id_emb], dim=-1)
        h = F.relu(self.input_proj(h))

        edge_attr = edge_weight.unsqueeze(-1) if edge_weight.numel() > 0 else None

        self.last_attention_weights = []
        for i, layer in enumerate(self.transformer_layers):
            if edge_index.shape[1] == 0:
                h = layer(h, edge_index, edge_attr=None)
            else:
                h, (attn_edge_index, attn_weights) = layer(
                    h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                self.last_attention_weights.append(
                    (attn_edge_index.detach(), attn_weights.detach())
                )
            if i < len(self.transformer_layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h

class GraphAutoencoderDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.node_stat_feature_dim),
        )

    def forward(self, z):
        return self.mlp(z)


class EdgeDecoder(nn.Module):
    def forward(self, z, edge_index):
        src, dst = edge_index
        logits = (z[src] * z[dst]).sum(dim=-1)
        return logits


class GraphTransformerAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphTransformerEncoder(config)
        self.decoder = GraphAutoencoderDecoder(config)
        self.edge_decoder = EdgeDecoder() if config.reconstruct_edges else None

    def forward(self, x_stats, id_idx, edge_index, edge_weight):
        z = self.encoder(x_stats, id_idx, edge_index, edge_weight)
        x_recon = self.decoder(z)

        edge_logits = None
        if self.edge_decoder is not None and edge_index.shape[1] > 0:
            edge_logits = self.edge_decoder(z, edge_index)

        return {
            "z": z,
            "x_recon": x_recon,
            "edge_logits": edge_logits,
        }

    def get_attention_weights(self):
        return self.encoder.last_attention_weights

    def compute_edge_logits_with_neg_sampling(self, z, pos_edge_index, num_nodes):
        if pos_edge_index.shape[1] == 0:
            return None, None

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=int(pos_edge_index.shape[1] * self.config.neg_sample_ratio),
            force_undirected=False,
        )

        pos_logits = self.edge_decoder(z, pos_edge_index)
        neg_logits = self.edge_decoder(z, neg_edge_index)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ], dim=0)

        return logits, labels

    def get_edge_reconstruction_logits(self, z, edge_index):
        if self.edge_decoder is None or edge_index.shape[1] == 0:
            return None
        return self.edge_decoder(z, edge_index)


def reconstruction_loss(outputs: dict, x_stats: torch.Tensor, edge_index: torch.Tensor,
                         config: ModelConfig, model: GraphTransformerAutoencoder = None) -> torch.Tensor:
    feature_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.2, 2.0, 1.0, 1.5, 2.5, 2.5],
        device=x_stats.device,
        dtype=x_stats.dtype,
    )
    node_residual = outputs["x_recon"] - x_stats
    node_loss = (node_residual.pow(2) * feature_weights).mean()

    if config.reconstruct_edges and outputs["edge_logits"] is not None and model is not None:
        logits, labels = model.compute_edge_logits_with_neg_sampling(
            outputs["z"], edge_index, x_stats.shape[0]
        )
        if logits is not None:
            edge_loss = F.binary_cross_entropy_with_logits(logits, labels)
            return node_loss + edge_loss

    return node_loss