#!/usr/bin/env python3
"""
Phase B - Step 3: Export graph statistics and save final graph.
Prints comprehensive summary, saves GraphML for potential Neo4j import.

Output: data/intermediate/mitograph.graphml
        data/intermediate/graph_summary.json
"""

import json
import pickle
import os
import networkx as nx


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    # Load graph
    pickle_path = os.path.join(intermediate_dir, 'mitograph_full.pkl')
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # --- Comprehensive statistics ---
    node_types = {}
    for node_id, data in G.nodes(data=True):
        nt = data.get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1
    
    edge_types = {}
    for _, _, data in G.edges(data=True):
        et = data.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1
    
    # Variant node statistics
    variant_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'Variant']
    clin_sig_dist = {}
    for _, d in variant_nodes:
        cs = d.get('clinical_significance', 'unknown')
        clin_sig_dist[cs] = clin_sig_dist.get(cs, 0) + 1
    
    # Degree statistics per node type
    degree_stats = {}
    for nt in node_types:
        nodes_of_type = [n for n, d in G.nodes(data=True) if d.get('node_type') == nt]
        degrees = [G.degree(n) for n in nodes_of_type]
        if degrees:
            degree_stats[nt] = {
                'min': min(degrees),
                'max': max(degrees),
                'mean': sum(degrees) / len(degrees),
            }
    
    # Phenotype connectivity
    pheno_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_type') == 'Phenotype']
    pheno_degrees = [(d.get('disease_name', n), G.degree(n)) for n, d in pheno_nodes]
    pheno_degrees.sort(key=lambda x: x[1], reverse=True)
    
    # Connected components
    n_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    
    summary = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
        'clinical_significance_distribution': clin_sig_dist,
        'degree_stats_by_node_type': degree_stats,
        'connected_components': n_components,
        'largest_component_size': len(largest_cc),
        'top_10_phenotypes_by_degree': pheno_degrees[:10],
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MITOGRAPH - Knowledge Graph Summary")
    print(f"{'='*60}")
    print(f"\nNodes: {summary['total_nodes']}")
    for nt, count in node_types.items():
        print(f"  {nt}: {count}")
    
    print(f"\nEdges: {summary['total_edges']}")
    for et, count in edge_types.items():
        print(f"  {et}: {count}")
    
    print(f"\nVariant clinical significance:")
    for cs, count in sorted(clin_sig_dist.items(), key=lambda x: -x[1]):
        print(f"  {cs}: {count}")
    
    print(f"\nDegree statistics by node type:")
    for nt, stats in degree_stats.items():
        print(f"  {nt}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")
    
    print(f"\nConnected components: {n_components}")
    print(f"Largest component: {len(largest_cc)} nodes")
    
    print(f"\nTop 10 phenotypes by connectivity:")
    for name, degree in pheno_degrees[:10]:
        print(f"  {name}: {degree} connections")
    
    # Save GraphML
    graphml_path = os.path.join(intermediate_dir, 'mitograph.graphml')
    nx.write_graphml(G, graphml_path)
    print(f"\nSaved GraphML to {graphml_path}")
    
    # Save summary
    summary_path = os.path.join(intermediate_dir, 'graph_summary.json')
    # Convert tuples to lists for JSON
    summary['top_10_phenotypes_by_degree'] = [
        {'name': name, 'degree': degree} for name, degree in summary['top_10_phenotypes_by_degree']
    ]
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

if __name__ == '__main__':
    main()
