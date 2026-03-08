#!/usr/bin/env python3
"""
Graph Attention Network (GATv2) link prediction model for heterogeneous graphs.

Architecture:
  Encoder: 2-layer HeteroConv (GATv2Conv per edge type, 4 attention heads)
  Decoder: Dot-product decoder for (Variant, ASSOCIATED_WITH, Phenotype) edges
  Loss:    Binary cross-entropy with hard negative mining

GATv2Conv learns attention weights per edge, dynamically deciding which
neighbor messages matter most. This prevents noisy KMER_SIMILARITY edges
from washing out strong biological signals from LOCATED_IN edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear


class MitoGraphEncoder(nn.Module):
    """
    Heterogeneous graph encoder using GATv2Conv per edge type.
    Multi-head attention (4 heads) learns which edges carry biological signal.
    """

    def __init__(self, metadata, hidden_dim=64, out_dim=32,
                 variant_in_dim=10, gene_in_dim=3, complex_in_dim=4,
                 phenotype_in_dim=64, heads=4):
        """
        Args:
            metadata: tuple (node_types, edge_types) from HeteroData.metadata()
            hidden_dim: hidden layer dimension (per head)
            out_dim: output embedding dimension
            *_in_dim: input feature dimensions for each node type
            heads: number of attention heads
        """
        super().__init__()

        self.heads = heads
        node_types, edge_types = metadata

        # Input projection layers (project each node type to hidden_dim)
        self.input_proj = nn.ModuleDict()
        in_dims = {
            'variant': variant_in_dim,
            'gene': gene_in_dim,
            'complex': complex_in_dim,
            'phenotype': phenotype_in_dim,
        }
        for nt in node_types:
            self.input_proj[nt] = Linear(in_dims.get(nt, hidden_dim), hidden_dim)

        # Layer 1: HeteroConv with GATv2Conv per edge type
        # GATv2Conv with heads=4 outputs hidden_dim (concat mode → hidden_dim * heads,
        # so we use hidden_dim // heads per head to keep total = hidden_dim)
        head_dim = hidden_dim // heads
        conv1_dict = {}
        for edge_type in edge_types:
            conv1_dict[edge_type] = GATv2Conv(
                (-1, -1), head_dim, heads=heads, concat=True, add_self_loops=False
            )
        self.conv1 = HeteroConv(conv1_dict, aggr='mean')

        # Layer 2: GATv2Conv → out_dim (single head for final embedding)
        conv2_dict = {}
        for edge_type in edge_types:
            conv2_dict[edge_type] = GATv2Conv(
                (-1, -1), out_dim, heads=1, concat=False, add_self_loops=False
            )
        self.conv2 = HeteroConv(conv2_dict, aggr='mean')

        self.dropout = nn.Dropout(0.3)

    def forward(self, x_dict, edge_index_dict):
        """
        Args:
            x_dict: dict[node_type] -> tensor of node features
            edge_index_dict: dict[edge_type] -> [2, num_edges] tensor

        Returns:
            dict[node_type] -> tensor of node embeddings (out_dim)
        """
        # Project inputs to hidden_dim
        h_dict = {}
        for nt, x in x_dict.items():
            h_dict[nt] = self.input_proj[nt](x)

        # Layer 1: multi-head attention
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {nt: F.elu(self.dropout(h)) for nt, h in h_dict.items()}

        # Layer 2: single-head → final embeddings
        h_dict = self.conv2(h_dict, edge_index_dict)

        return h_dict


class DotProductDecoder(nn.Module):
    """
    Dot-product decoder: score(u, v) = u^T v.
    For predicting (Variant, ASSOCIATED_WITH, Phenotype) links.
    """

    def forward(self, z_src, z_dst, edge_index):
        """
        Args:
            z_src: source node embeddings [num_src_nodes, dim]
            z_dst: destination node embeddings [num_dst_nodes, dim]
            edge_index: [2, num_edges] index into src/dst

        Returns:
            scores: [num_edges] logits
        """
        src_emb = z_src[edge_index[0]]
        dst_emb = z_dst[edge_index[1]]
        return (src_emb * dst_emb).sum(dim=-1)


class MitoGraphLinkPredictor(nn.Module):
    """
    Full model: GATv2 Encoder + Dot-Product Decoder for link prediction.
    """

    def __init__(self, metadata, hidden_dim=64, out_dim=32,
                 variant_in_dim=10, gene_in_dim=3, complex_in_dim=4,
                 phenotype_in_dim=64, heads=4):
        super().__init__()
        self.encoder = MitoGraphEncoder(
            metadata, hidden_dim, out_dim,
            variant_in_dim, gene_in_dim, complex_in_dim, phenotype_in_dim,
            heads=heads
        )
        self.decoder = DotProductDecoder()

    def forward(self, x_dict, edge_index_dict):
        """Encode all nodes → embeddings dict."""
        return self.encoder(x_dict, edge_index_dict)

    def predict_links(self, z_dict, edge_index):
        """
        Predict link scores for (variant, phenotype) pairs.
        Args:
            z_dict: node embeddings from encoder
            edge_index: [2, num_edges] with [variant_indices, phenotype_indices]
        Returns:
            logits: [num_edges]
        """
        return self.decoder(z_dict['variant'], z_dict['phenotype'], edge_index)
