#!/usr/bin/env python3
"""
Phase B - Step 1: Build the Knowledge Graph in NetworkX.
Creates a heterogeneous graph with:
  Nodes: Variant, Gene, Complex, Phenotype
  Edges: LOCATED_IN, PART_OF, ASSOCIATED_WITH

Output: data/intermediate/mitograph_base.graphml (before k-mer edges)
        data/intermediate/graph_stats.json
"""

import pandas as pd
import networkx as nx
import json
import os
import re


def parse_phenotypes(phenotype_str):
    """Parse phenotype string from ClinVar/MITOMAP into individual diseases."""
    if pd.isna(phenotype_str) or str(phenotype_str).strip() == '':
        return []
    
    diseases = []
    # Split by pipe
    for part in str(phenotype_str).split('|'):
        part = part.strip()
        if part and part.lower() not in ('not provided', 'not specified', 'na', ''):
            # Clean up
            part = re.sub(r'\s+', ' ', part)
            diseases.append(part)
    return diseases


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    # --- Load data ---
    variants = pd.read_csv(os.path.join(intermediate_dir, 'merged_variants.csv'))
    genes = pd.read_csv(os.path.join(intermediate_dir, 'genes.csv'))
    complexes = pd.read_csv(os.path.join(intermediate_dir, 'gene_complex_mapping.csv'))
    
    print(f"Loaded: {len(variants)} variants, {len(genes)} genes")
    
    # --- Build graph ---
    G = nx.Graph()
    
    # 1. Add Gene nodes
    for _, row in genes.iterrows():
        G.add_node(
            f"gene_{row['gene_name']}",
            node_type='Gene',
            name=row['gene_name'],
            biotype=row['biotype'],
            start=int(row['start']),
            end=int(row['end']),
            strand=row['strand']
        )
    print(f"Added {len(genes)} Gene nodes")
    
    # 2. Add Complex nodes
    complex_ids = complexes[complexes['complex'] != 'none']['complex'].unique()
    for c_id in complex_ids:
        G.add_node(
            f"complex_{c_id}",
            node_type='Complex',
            complex_id=c_id
        )
    print(f"Added {len(complex_ids)} Complex nodes: {sorted(complex_ids)}")
    
    # 3. Add Gene -> Complex edges (PART_OF)
    part_of_count = 0
    for _, row in complexes[complexes['complex'] != 'none'].iterrows():
        gene_id = f"gene_{row['gene_name']}"
        complex_id = f"complex_{row['complex']}"
        if G.has_node(gene_id) and G.has_node(complex_id):
            G.add_edge(gene_id, complex_id, edge_type='PART_OF')
            part_of_count += 1
    print(f"Added {part_of_count} PART_OF edges (Gene -> Complex)")
    
    # 4. Add Variant nodes and LOCATED_IN edges
    variant_count = 0
    located_in_count = 0
    phenotype_set = set()
    
    for _, var in variants.iterrows():
        var_id = f"var_{int(var['pos'])}_{var['ref']}_{var['alt']}"
        
        # Add variant node
        attrs = {
            'node_type': 'Variant',
            'pos': int(var['pos']),
            'ref': str(var['ref']),
            'alt': str(var['alt']),
            'clinical_significance': str(var.get('clinical_significance', 'unknown')),
            'phylop_score': float(var['phylop_score']) if pd.notna(var.get('phylop_score')) else float('nan'),
        }
        
        # Add optional scores
        if pd.notna(var.get('apogee_score')):
            attrs['apogee_score'] = float(var['apogee_score'])
        if pd.notna(var.get('mitotip_score')):
            attrs['mitotip_score'] = float(var['mitotip_score'])
        
        G.add_node(var_id, **attrs)
        variant_count += 1
        
        # LOCATED_IN: check which gene(s) this variant falls in
        for _, gene in genes.iterrows():
            if gene['start'] <= var['pos'] <= gene['end']:
                G.add_edge(var_id, f"gene_{gene['gene_name']}", edge_type='LOCATED_IN')
                located_in_count += 1
        
        # Collect phenotypes for ASSOCIATED_WITH edges
        # From ClinVar
        clinvar_phenos = parse_phenotypes(var.get('phenotype_list'))
        # From MITOMAP
        mitomap_phenos = parse_phenotypes(var.get('mitomap_disease'))
        
        all_phenos = set(clinvar_phenos + mitomap_phenos)
        phenotype_set.update(all_phenos)
    
    print(f"Added {variant_count} Variant nodes")
    print(f"Added {located_in_count} LOCATED_IN edges (Variant -> Gene)")
    
    # 5. Add Phenotype nodes
    for pheno in phenotype_set:
        pheno_id = f"pheno_{pheno.replace(' ', '_').replace('/', '_')[:80]}"
        G.add_node(pheno_id, node_type='Phenotype', disease_name=pheno)
    print(f"Added {len(phenotype_set)} Phenotype nodes")
    
    # 6. Add ASSOCIATED_WITH edges (Variant -> Phenotype)
    # Only for variants with confirmed associations (Pathogenic/Likely pathogenic from ClinVar,
    # or Confirmed from MITOMAP)
    assoc_count = 0
    for _, var in variants.iterrows():
        var_id = f"var_{int(var['pos'])}_{var['ref']}_{var['alt']}"
        
        clin_sig = str(var.get('clinical_significance', '')).lower()
        mitomap_status = str(var.get('mitomap_status', '')).lower()
        
        # Determine if this variant has confirmed associations
        has_confirmed = (
            'pathogenic' in clin_sig or 
            'cfrm' in mitomap_status or
            'confirmed' in mitomap_status
        )
        
        if not has_confirmed:
            continue
        
        all_phenos = set(
            parse_phenotypes(var.get('phenotype_list')) + 
            parse_phenotypes(var.get('mitomap_disease'))
        )
        
        for pheno in all_phenos:
            pheno_id = f"pheno_{pheno.replace(' ', '_').replace('/', '_')[:80]}"
            if G.has_node(pheno_id):
                G.add_edge(var_id, pheno_id, edge_type='ASSOCIATED_WITH')
                assoc_count += 1
    
    print(f"Added {assoc_count} ASSOCIATED_WITH edges (Variant -> Phenotype)")
    
    # --- Graph statistics ---
    node_types = {}
    for _, data in G.nodes(data=True):
        nt = data.get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1
    
    edge_types = {}
    for _, _, data in G.edges(data=True):
        et = data.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1
    
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
        'is_connected': nx.is_connected(G),
        'n_connected_components': nx.number_connected_components(G),
    }
    
    print(f"\n--- Graph Statistics ---")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Node types: {json.dumps(node_types, indent=2)}")
    print(f"Edge types: {json.dumps(edge_types, indent=2)}")
    print(f"Connected: {stats['is_connected']}")
    print(f"Connected components: {stats['n_connected_components']}")
    
    # Save graph
    graph_path = os.path.join(intermediate_dir, 'mitograph_base.graphml')
    nx.write_graphml(G, graph_path)
    print(f"\nSaved graph to {graph_path}")
    
    # Save stats
    stats_path = os.path.join(intermediate_dir, 'graph_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")
    
    # Save graph as pickle for faster reloading
    import pickle
    pickle_path = os.path.join(intermediate_dir, 'mitograph_base.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved pickle to {pickle_path}")

if __name__ == '__main__':
    main()
