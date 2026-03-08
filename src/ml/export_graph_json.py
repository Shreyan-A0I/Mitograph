#!/usr/bin/env python3
"""Export a filtered network graph as JSON for the Next.js frontend.

Full graph: 4,288 nodes, 11,499 edges (too many for browser force graph).

Filtered graph shows:
  - 4 Complexes (root nodes)
  - 37 Genes (connected to complexes)
  - Top pathogenic + flagged VUS variants 
  - Top phenotypes connected to those variants

Output: frontend/public/data/network_graph.json
"""

import json, os
from collections import Counter
import pandas as pd
import torch

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

    var_labels = meta['var_labels']
    idx_to_var = meta['idx_to_var']
    pheno_names = meta['pheno_names']

    nodes = []
    links = []
    node_ids = set()

    # === COMPLEXES ===
    for name in meta['complex_to_idx']:
        nid = f'C:{name}'
        nodes.append({'id': nid, 'label': name, 'type': 'Complex', 'size': 24})
        node_ids.add(nid)

    # === GENES ===
    gene_names = list(meta['gene_to_idx'].keys())
    for name in gene_names:
        nid = f'G:{name}'
        nodes.append({'id': nid, 'label': name, 'type': 'Gene', 'size': 14})
        node_ids.add(nid)

    # Gene → Complex edges
    gc_path = os.path.join(intermediate_dir, 'gene_complex_mapping.csv')
    gc_df = pd.read_csv(gc_path)
    for _, row in gc_df.iterrows():
        g_id = f"G:{row['gene_name']}"
        cplx = str(row['complex'])
        if cplx == 'none' or pd.isna(row['complex']):
            continue
        c_id = f"C:{cplx}"
        if g_id in node_ids and c_id in node_ids:
            links.append({'source': g_id, 'target': c_id, 'type': 'PART_OF'})

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
    # Add top flagged VUS
    top_variants.update(list(flagged_indices)[:50])

    for v_idx in top_variants:
        var_id = idx_to_var.get(str(v_idx), f'var_{v_idx}')
        is_flagged = var_id in flagged_var_ids
        vtype = 'Flagged_VUS' if is_flagged else 'Pathogenic'
        nodes.append({
            'id': f'V:{var_id}', 'label': var_id,
            'type': vtype, 'size': 8 if is_flagged else 6
        })
        node_ids.add(f'V:{var_id}')

    # Variant → Gene edges
    for i in range(loc_ei.shape[1]):
        v_idx = loc_ei[0, i].item()
        g_idx = loc_ei[1, i].item()
        if v_idx in top_variants and g_idx < len(gene_names):
            v_node = f'V:{idx_to_var.get(str(v_idx), f"var_{v_idx}")}'
            g_node = f'G:{gene_names[g_idx]}'
            if v_node in node_ids and g_node in node_ids:
                links.append({'source': v_node, 'target': g_node, 'type': 'LOCATED_IN'})

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
        short = p_name[:45] + '…' if len(p_name) > 45 else p_name
        nid = f'P:{p_name}'
        nodes.append({'id': nid, 'label': short, 'type': 'Phenotype', 'size': 10})
        node_ids.add(nid)

    # Variant → Phenotype edges
    for i in range(assoc_ei.shape[1]):
        v_idx = assoc_ei[0, i].item()
        p_idx = assoc_ei[1, i].item()
        if v_idx in top_variants and p_idx in top_phenos:
            var_id = idx_to_var.get(str(v_idx), f'var_{v_idx}')
            p_name = pheno_names.get(str(p_idx), f'Phenotype_{p_idx}')
            links.append({
                'source': f'V:{var_id}', 'target': f'P:{p_name}',
                'type': 'ASSOCIATED_WITH'
            })

    # === SAVE ===
    graph_json = {'nodes': nodes, 'links': links}
    type_counts = Counter(n['type'] for n in nodes)
    link_counts = Counter(l['type'] for l in links)

    print(f"Network graph: {len(nodes)} nodes, {len(links)} links")
    print(f"  Nodes: {dict(type_counts)}")
    print(f"  Links: {dict(link_counts)}")

    out_path = os.path.join(export_dir, 'network_graph.json')
    with open(out_path, 'w') as f:
        json.dump(graph_json, f)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
