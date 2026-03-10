#!/usr/bin/env python3
"""Export a filtered network graph as JSON for the Next.js frontend.

Full graph: 4,288 nodes, 11,499 edges (too many for browser force graph).

Filtered graph shows:
  - 4 Complexes (root nodes)
  - 37 Genes (connected to complexes)
  - Top pathogenic + flagged VUS variants 
  - Top phenotypes connected to those variants
  - Edge attention weights from GATv2Conv final layer

Output: frontend/public/data/network_graph.json
"""

import json, os, sys, math, pickle
from collections import Counter
import pandas as pd
import numpy as np
import torch

# Add current dir to path for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MitoGraphLinkPredictor


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    results_dir = os.path.join(project_dir, 'data', 'results')
    export_dir = os.path.join(project_dir, 'frontend', 'public', 'data')
    os.makedirs(export_dir, exist_ok=True)

    # Load data
    data = torch.load(os.path.join(intermediate_dir, 'hetero_data.pt'), weights_only=False)
    with open(os.path.join(intermediate_dir, 'graph_metadata.json')) as f:
        meta = json.load(f)

    # Load NetworkX graph for raw node attributes (ref/alt/pos etc.)
    with open(os.path.join(intermediate_dir, 'mitograph_full.pkl'), 'rb') as f:
        G = pickle.load(f)

    # Build raw node data lookup
    raw_node_data = {}
    for nid, ndata in G.nodes(data=True):
        raw_node_data[nid] = ndata

    var_labels = meta['var_labels']
    idx_to_var = meta['idx_to_var']
    pheno_names = meta['pheno_names']

    # ================================================================
    # ATTENTION WEIGHT EXTRACTION
    # ================================================================
    print("Extracting GATv2Conv attention weights...")

    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Read model config
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    config = metrics.get('model_config', {})
    hidden_dim = config.get('hidden_dim', 64)
    heads = config.get('attention_heads', 4)

    pheno_in_dim = data['phenotype'].x.shape[1]
    model = MitoGraphLinkPredictor(
        metadata=data.metadata(),
        hidden_dim=hidden_dim, out_dim=32,
        variant_in_dim=data['variant'].x.shape[1],
        gene_in_dim=data['gene'].x.shape[1],
        complex_in_dim=data['complex'].x.shape[1],
        phenotype_in_dim=pheno_in_dim,
        heads=heads,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ================================================================
    # FEATURE IMPORTANCE EXTRACTION
    # ================================================================
    print("Extracting feature linear projection weights...")
    weights = model.encoder.input_proj['variant'].weight.detach().abs().mean(dim=0).numpy()
    total_weight = weights.sum()
    pcts = [(float(w) / float(total_weight)) * 100 for w in weights]
    
    # 14D pure biological features
    feature_names = [
        "PhyloP (Conservation)",
        "Position X (sin)",
        "Position Y (cos)",
        "Is Transition",
        "Is Transversion",
        "Is Indel",
        "Ref: A", "Ref: C", "Ref: G", "Ref: T",
        "Alt: A", "Alt: C", "Alt: G", "Alt: T"
    ]
    
    importance_data = [{"feature": name, "importance": pct} for name, pct in zip(feature_names, pcts)]
    # Sort descending
    importance_data.sort(key=lambda x: x["importance"], reverse=True)
    
    feat_out_path = os.path.join(export_dir, 'feature_importance.json')
    with open(feat_out_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    print(f"Exported feature importance to {feat_out_path}")

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

    with torch.no_grad():
        _, attention_dict = model.forward_with_attention(x_dict, edge_index_dict)

    # Build attention lookup: (edge_type_str, src_idx, dst_idx) -> mean alpha
    attention_lookup = {}
    for edge_type, (ei, alpha) in attention_dict.items():
        src_type, rel, dst_type = edge_type
        key_prefix = rel.upper()  # e.g. 'LOCATED_IN'
        # alpha shape: [num_edges, heads] -> mean across heads
        mean_alpha = alpha.mean(dim=-1).numpy()
        for i in range(ei.shape[1]):
            s, d = ei[0, i].item(), ei[1, i].item()
            attention_lookup[(key_prefix, s, d)] = float(mean_alpha[i])
    print(f"  Extracted {len(attention_lookup)} edge attention values")

    # ================================================================
    # BUILD FILTERED GRAPH
    # ================================================================
    nodes = []
    links = []
    node_ids = set()

    # === COMPLEXES ===
    for name in meta['complex_to_idx']:
        nid = f'C:{name}'
        nodes.append({
            'id': nid, 'label': name, 'type': 'Complex', 'size': 24,
            'features': {'complex_id': name}
        })
        node_ids.add(nid)

    # === GENES ===
    gene_names = list(meta['gene_to_idx'].keys())
    for name in gene_names:
        raw = raw_node_data.get(name, {})
        biotype = raw.get('biotype', 'unknown')
        start = raw.get('start', '')
        end = raw.get('end', '')
        nid = f'G:{name}'
        nodes.append({
            'id': nid, 'label': name, 'type': 'Gene', 'size': 14,
            'features': {
                'biotype': biotype,
                'genome_range': f"{start}-{end}" if start else 'unknown',
            }
        })
        node_ids.add(nid)

    # Gene -> Complex edges
    gc_path = os.path.join(intermediate_dir, 'gene_complex_mapping.csv')
    gc_df = pd.read_csv(gc_path)
    for _, row in gc_df.iterrows():
        g_id = f"G:{row['gene_name']}"
        cplx = str(row['complex'])
        if cplx == 'none' or pd.isna(row['complex']):
            continue
        c_id = f"C:{cplx}"
        if g_id in node_ids and c_id in node_ids:
            # Lookup attention for PART_OF edges
            g_idx = meta['gene_to_idx'].get(row['gene_name'])
            c_idx = meta['complex_to_idx'].get(cplx)
            att = attention_lookup.get(('PART_OF', g_idx, c_idx), 0.5)
            links.append({
                'source': g_id, 'target': c_id,
                'type': 'PART_OF', 'attention': round(att, 4)
            })

    # === VARIANTS (filtered) ===
    assoc_ei = data[('variant', 'associated_with', 'phenotype')].edge_index
    loc_ei = data[('variant', 'located_in', 'gene')].edge_index

    # Pathogenic indices
    path_indices = set()
    for idx_str, label in var_labels.items():
        if 'pathogenic' in label.lower():
            path_indices.add(int(idx_str))

    # Flagged VUS
    with open(os.path.join(results_dir, 'vus_summary.json')) as f:
        vus_summary = json.load(f)
    flagged_var_ids = {v['variant_id'] for v in vus_summary.get('vus_flagged', [])}
    flagged_indices = set()
    for idx_str, var_id in idx_to_var.items():
        if var_id in flagged_var_ids:
            flagged_indices.add(int(idx_str))

    # Pathogenic variants with ASSOCIATED_WITH edges
    interesting = set()
    for i in range(assoc_ei.shape[1]):
        v_idx = assoc_ei[0, i].item()
        if v_idx in path_indices:
            interesting.add(v_idx)

    # Top pathogenic by edge count
    var_edge_count = Counter()
    for i in range(assoc_ei.shape[1]):
        v_idx = assoc_ei[0, i].item()
        if v_idx in interesting:
            var_edge_count[v_idx] += 1

    top_variants = set(v for v, _ in var_edge_count.most_common(60))
    top_variants.update(list(flagged_indices)[:50])

    # Variant feature tensor for quick access
    var_feats = data['variant'].x.numpy()

    for v_idx in top_variants:
        var_id = idx_to_var.get(str(v_idx), f'var_{v_idx}')
        is_flagged = var_id in flagged_var_ids
        vtype = 'Flagged_VUS' if is_flagged else 'Pathogenic'

        # Extract raw node data for this variant
        raw = raw_node_data.get(var_id, {})
        feats = var_feats[v_idx]

        clin_sig = raw.get('clinical_significance', 'Unknown')
        pos = raw.get('pos', int(round(
            math.atan2(feats[1], feats[2]) / (2 * math.pi) * 16569
        )) if feats[2] != 0 else 0)

        is_transition = bool(feats[3] > 0.5)
        is_transversion = bool(feats[4] > 0.5)
        is_indel = bool(feats[5] > 0.5)
        mut_type = "Indel"
        if is_transition: mut_type = "Transition"
        elif is_transversion: mut_type = "Transversion"

        features = {
            'position': int(pos) if pos else 0,
            'ref': raw.get('ref', '?'),
            'alt': raw.get('alt', '?'),
            'clinical_significance': clin_sig,
            'phylop': round(float(feats[0]), 4),
            'mutation_type': mut_type
        }

        nodes.append({
            'id': f'V:{var_id}', 'label': var_id,
            'type': vtype, 'size': 8 if is_flagged else 6,
            'features': features,
        })
        node_ids.add(f'V:{var_id}')

    # Variant -> Gene edges
    for i in range(loc_ei.shape[1]):
        v_idx = loc_ei[0, i].item()
        g_idx = loc_ei[1, i].item()
        if v_idx in top_variants and g_idx < len(gene_names):
            v_node = f'V:{idx_to_var.get(str(v_idx), f"var_{v_idx}")}'
            g_node = f'G:{gene_names[g_idx]}'
            if v_node in node_ids and g_node in node_ids:
                att = attention_lookup.get(('LOCATED_IN', v_idx, g_idx), 0.5)
                links.append({
                    'source': v_node, 'target': g_node,
                    'type': 'LOCATED_IN', 'attention': round(att, 4),
                })

    # === PHENOTYPES (top connected) ===
    pheno_connected = Counter()
    for i in range(assoc_ei.shape[1]):
        v_idx = assoc_ei[0, i].item()
        p_idx = assoc_ei[1, i].item()
        if v_idx in top_variants:
            pheno_connected[p_idx] += 1

    top_phenos = set(p for p, _ in pheno_connected.most_common(25))
    for p_idx in top_phenos:
        p_name = pheno_names.get(str(p_idx), f'Phenotype_{p_idx}')
        short = p_name[:45] + '...' if len(p_name) > 45 else p_name
        nid = f'P:{p_name}'
        nodes.append({
            'id': nid, 'label': short, 'type': 'Phenotype', 'size': 10,
            'features': {
                'disease_name': p_name,
                'connected_variants': pheno_connected[p_idx],
            }
        })
        node_ids.add(nid)

    # Variant -> Phenotype edges
    for i in range(assoc_ei.shape[1]):
        v_idx = assoc_ei[0, i].item()
        p_idx = assoc_ei[1, i].item()
        if v_idx in top_variants and p_idx in top_phenos:
            var_id = idx_to_var.get(str(v_idx), f'var_{v_idx}')
            p_name = pheno_names.get(str(p_idx), f'Phenotype_{p_idx}')
            att = attention_lookup.get(('ASSOCIATED_WITH', v_idx, p_idx), 0.5)
            links.append({
                'source': f'V:{var_id}', 'target': f'P:{p_name}',
                'type': 'ASSOCIATED_WITH', 'attention': round(att, 4),
            })

    # === SAVE ===
    graph_json = {'nodes': nodes, 'links': links}
    type_counts = Counter(n['type'] for n in nodes)
    link_counts = Counter(l['type'] for l in links)

    # Attention statistics
    att_values = [l['attention'] for l in links if 'attention' in l]
    print(f"\nNetwork graph: {len(nodes)} nodes, {len(links)} links")
    print(f"  Nodes: {dict(type_counts)}")
    print(f"  Links: {dict(link_counts)}")
    if att_values:
        print(f"  Attention: min={min(att_values):.4f}, max={max(att_values):.4f}, "
              f"mean={np.mean(att_values):.4f}")

    out_path = os.path.join(export_dir, 'network_graph.json')
    with open(out_path, 'w') as f:
        json.dump(graph_json, f)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
