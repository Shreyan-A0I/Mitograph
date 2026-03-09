#!/usr/bin/env python3
"""
Phase C - Step 1: Convert NetworkX graph → PyG HeteroData.

Encodes node features:
  Variant: [phylop_score, is_pathogenic, is_benign, is_vus, sin_pos, cos_pos,
            apogee_score, mitotip_score]
  Gene:    one-hot [tRNA, rRNA, protein_coding]
  Complex: one-hot [I, III, IV, V]
  Phenotype: identity (index-based embedding)

Encodes edge indices for each relation type:
  (Variant, LOCATED_IN, Gene)
  (Gene, PART_OF, Complex)
  (Variant, ASSOCIATED_WITH, Phenotype)
  (Variant, KMER_SIMILARITY, Variant)

Output: data/intermediate/hetero_data.pt
"""

import pickle
import os
import math
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


MT_GENOME_LENGTH = 16569


def circular_positional_encoding(pos):
    """
    Encode a linear position on the circular mitochondrial genome as a 2D
    trigonometric embedding: (sin(2π·pos/16569), cos(2π·pos/16569)).
    This preserves the fact that position 16569 is adjacent to position 1.
    """
    theta = 2.0 * math.pi * pos / MT_GENOME_LENGTH
    return math.sin(theta), math.cos(theta)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')

    # --- Load graph ---
    pickle_path = os.path.join(intermediate_dir, 'mitograph_full.pkl')
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Separate nodes by type ---
    variant_nodes = []
    gene_nodes = []
    complex_nodes = []
    phenotype_nodes = []

    for node_id, data in G.nodes(data=True):
        nt = data.get('node_type', 'unknown')
        if nt == 'Variant':
            variant_nodes.append((node_id, data))
        elif nt == 'Gene':
            gene_nodes.append((node_id, data))
        elif nt == 'Complex':
            complex_nodes.append((node_id, data))
        elif nt == 'Phenotype':
            phenotype_nodes.append((node_id, data))

    print(f"Variants: {len(variant_nodes)}, Genes: {len(gene_nodes)}, "
          f"Complexes: {len(complex_nodes)}, Phenotypes: {len(phenotype_nodes)}")

    # --- Build node ID → index mappings ---
    var_to_idx = {nid: i for i, (nid, _) in enumerate(variant_nodes)}
    gene_to_idx = {nid: i for i, (nid, _) in enumerate(gene_nodes)}
    complex_to_idx = {nid: i for i, (nid, _) in enumerate(complex_nodes)}
    pheno_to_idx = {nid: i for i, (nid, _) in enumerate(phenotype_nodes)}

    # --- Encode Variant features ---
    # 14D Pure Biological Feature Vector:
    # [0] phylop_score
    # [1] sin_pos
    # [2] cos_pos
    # [3] is_transition
    # [4] is_transversion
    # [5] is_indel
    # [6-9] ref_A, ref_C, ref_G, ref_T (one-hot)
    # [10-13] alt_A, alt_C, alt_G, alt_T (one-hot)
    n_var_feats = 14
    var_features = np.zeros((len(variant_nodes), n_var_feats), dtype=np.float32)

    # Track which variants have missing phylop for imputation
    phylop_values = []
    phylop_missing_indices = []

    for i, (nid, data) in enumerate(variant_nodes):
        # 0: PhyloP score (may be NaN — will impute later)
        phylop = data.get('phylop_score', float('nan'))
        if not math.isnan(phylop):
            var_features[i, 0] = phylop
            phylop_values.append(phylop)
        else:
            phylop_missing_indices.append(i)

        # 1-2: Circular positional encoding
        pos = data.get('pos', 1)
        sin_pos, cos_pos = circular_positional_encoding(pos)
        var_features[i, 1] = sin_pos
        var_features[i, 2] = cos_pos

        # 3-5: Mutation type
        ref = str(data.get('ref', '')).upper()
        alt = str(data.get('alt', '')).upper()
        
        is_transition = 0.0
        is_transversion = 0.0
        is_indel = 0.0

        if len(ref) == 1 and len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT':
            s = {ref, alt}
            if s == {'A', 'G'} or s == {'C', 'T'}:
                is_transition = 1.0
            else:
                is_transversion = 1.0
        else:
            is_indel = 1.0

        var_features[i, 3] = is_transition
        var_features[i, 4] = is_transversion
        var_features[i, 5] = is_indel

        # 6-9: One-hot Ref
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        if len(ref) == 1 and ref in base_map:
            var_features[i, 6 + base_map[ref]] = 1.0
            
        # 10-13: One-hot Alt
        if len(alt) == 1 and alt in base_map:
            var_features[i, 10 + base_map[alt]] = 1.0

    # Impute missing PhyloP with median (biologically more defensible than 0.0)
    if phylop_values and phylop_missing_indices:
        phylop_median = float(np.median(phylop_values))
        for idx in phylop_missing_indices:
            var_features[idx, 0] = phylop_median
        print(f"PhyloP: imputed {len(phylop_missing_indices)} missing values "
              f"with median={phylop_median:.4f}")
    print(f"PhyloP coverage: {len(phylop_values)}/{len(variant_nodes)} "
          f"({100*len(phylop_values)/len(variant_nodes):.1f}%)")

    # --- Encode Gene features ---
    # One-hot: [tRNA, rRNA, protein_coding]
    biotype_map = {'tRNA': 0, 'rRNA': 1, 'protein_coding': 2}
    gene_features = np.zeros((len(gene_nodes), 3), dtype=np.float32)
    for i, (nid, data) in enumerate(gene_nodes):
        biotype = data.get('biotype', '')
        if biotype in biotype_map:
            gene_features[i, biotype_map[biotype]] = 1.0

    # --- Encode Complex features ---
    # One-hot: [I, III, IV, V]
    complex_map = {'I': 0, 'III': 1, 'IV': 2, 'V': 3}
    complex_features = np.zeros((len(complex_nodes), 4), dtype=np.float32)
    for i, (nid, data) in enumerate(complex_nodes):
        cid = data.get('complex_id', '')
        if cid in complex_map:
            complex_features[i, complex_map[cid]] = 1.0

    # --- Encode Phenotype features ---
    # Identity feature (index-based, will be learned via embedding layer)
    pheno_features = np.eye(len(phenotype_nodes), dtype=np.float32)
    # If too many phenotypes, use a smaller identity
    if len(phenotype_nodes) > 256:
        # Use random projection to smaller dimension
        np.random.seed(42)
        proj = np.random.randn(len(phenotype_nodes), 64).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True)
        pheno_features = proj

    # --- Build edge indices ---
    # (Variant, LOCATED_IN, Gene) — bidirectional
    located_in_src, located_in_dst = [], []
    # (Gene, PART_OF, Complex)
    part_of_src, part_of_dst = [], []
    # (Variant, ASSOCIATED_WITH, Phenotype)
    assoc_src, assoc_dst = [], []
    # (Variant, KMER_SIMILARITY, Variant)
    kmer_src, kmer_dst = [], []

    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', '')

        if edge_type == 'LOCATED_IN':
            if u in var_to_idx and v in gene_to_idx:
                located_in_src.append(var_to_idx[u])
                located_in_dst.append(gene_to_idx[v])
            elif v in var_to_idx and u in gene_to_idx:
                located_in_src.append(var_to_idx[v])
                located_in_dst.append(gene_to_idx[u])

        elif edge_type == 'PART_OF':
            if u in gene_to_idx and v in complex_to_idx:
                part_of_src.append(gene_to_idx[u])
                part_of_dst.append(complex_to_idx[v])
            elif v in gene_to_idx and u in complex_to_idx:
                part_of_src.append(gene_to_idx[v])
                part_of_dst.append(complex_to_idx[u])

        elif edge_type == 'ASSOCIATED_WITH':
            if u in var_to_idx and v in pheno_to_idx:
                assoc_src.append(var_to_idx[u])
                assoc_dst.append(pheno_to_idx[v])
            elif v in var_to_idx and u in pheno_to_idx:
                assoc_src.append(var_to_idx[v])
                assoc_dst.append(pheno_to_idx[u])

        elif edge_type == 'KMER_SIMILARITY':
            if u in var_to_idx and v in var_to_idx:
                kmer_src.append(var_to_idx[u])
                kmer_dst.append(var_to_idx[v])

    # --- Build HeteroData ---
    hetero_data = HeteroData()

    # Node features
    hetero_data['variant'].x = torch.tensor(var_features)
    hetero_data['gene'].x = torch.tensor(gene_features)
    hetero_data['complex'].x = torch.tensor(complex_features)
    hetero_data['phenotype'].x = torch.tensor(pheno_features)

    # Edge indices (PyG uses [2, num_edges] format)
    if located_in_src:
        hetero_data['variant', 'located_in', 'gene'].edge_index = torch.tensor(
            [located_in_src, located_in_dst], dtype=torch.long)
        # Reverse edges for message passing
        hetero_data['gene', 'rev_located_in', 'variant'].edge_index = torch.tensor(
            [located_in_dst, located_in_src], dtype=torch.long)

    if part_of_src:
        hetero_data['gene', 'part_of', 'complex'].edge_index = torch.tensor(
            [part_of_src, part_of_dst], dtype=torch.long)
        hetero_data['complex', 'rev_part_of', 'gene'].edge_index = torch.tensor(
            [part_of_dst, part_of_src], dtype=torch.long)

    if assoc_src:
        hetero_data['variant', 'associated_with', 'phenotype'].edge_index = torch.tensor(
            [assoc_src, assoc_dst], dtype=torch.long)
        hetero_data['phenotype', 'rev_associated_with', 'variant'].edge_index = torch.tensor(
            [assoc_dst, assoc_src], dtype=torch.long)

    if kmer_src:
        hetero_data['variant', 'kmer_similar', 'variant'].edge_index = torch.tensor(
            [kmer_src, kmer_dst], dtype=torch.long)

    # --- Save metadata for later use ---
    metadata = {
        'var_to_idx': {k: int(v) for k, v in var_to_idx.items()},
        'gene_to_idx': {k: int(v) for k, v in gene_to_idx.items()},
        'complex_to_idx': {k: int(v) for k, v in complex_to_idx.items()},
        'pheno_to_idx': {k: int(v) for k, v in pheno_to_idx.items()},
        'idx_to_var': {int(v): k for k, v in var_to_idx.items()},
        'idx_to_pheno': {int(v): k for k, v in pheno_to_idx.items()},
    }

    # Save variant clinical significance labels for splitting
    var_labels = {}
    for nid, data in variant_nodes:
        var_labels[var_to_idx[nid]] = str(data.get('clinical_significance', 'unknown'))
    metadata['var_labels'] = var_labels

    # Save phenotype names
    pheno_names = {}
    for nid, data in phenotype_nodes:
        pheno_names[pheno_to_idx[nid]] = data.get('disease_name', nid)
    metadata['pheno_names'] = pheno_names

    # --- Print summary ---
    print(f"\n--- HeteroData Summary ---")
    print(hetero_data)
    print(f"\nNode types and feature dims:")
    for node_type in hetero_data.node_types:
        print(f"  {node_type}: {hetero_data[node_type].x.shape}")
    print(f"\nEdge types and counts:")
    for edge_type in hetero_data.edge_types:
        ei = hetero_data[edge_type].edge_index
        print(f"  {edge_type}: {ei.shape[1]} edges")

    print(f"\nASSOCIATED_WITH edges (prediction target): {len(assoc_src)}")
    print(f"VUS variants: {sum(1 for v in var_labels.values() if 'vus' in v.lower() or 'uncertain' in v.lower())}")

    # --- Save ---
    data_path = os.path.join(intermediate_dir, 'hetero_data.pt')
    torch.save(hetero_data, data_path)
    print(f"\nSaved HeteroData to {data_path}")

    meta_path = os.path.join(intermediate_dir, 'graph_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {meta_path}")


if __name__ == '__main__':
    main()
