#!/usr/bin/env python3
"""
Phase B - Step 2: Compute k-mer similarity edges between variants.
Uses ±20bp window around each variant position on the rCRS mitochondrial genome.
Computes 4-mer frequency vectors and connects variants with high cosine similarity.

The circular mtDNA genome is handled properly (wrapping around position 16569->1).

Output: data/intermediate/kmer_edges.csv
        Updates the graph with KMER_SIMILARITY edges
"""

import pandas as pd
import numpy as np
from itertools import product
from collections import Counter
from scipy.spatial.distance import cosine
import pickle
import os


def load_fasta_sequence(fasta_path):
    """Load FASTA sequence, return as a single string (uppercase)."""
    seq_parts = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                continue
            seq_parts.append(line.strip().upper())
    return ''.join(seq_parts)


def get_circular_window(sequence, pos, window_size=20):
    """
    Extract a window around position `pos` (1-indexed) from a circular genome.
    window_size is the number of bases on each side (total window = 2*window_size + 1).
    Handles wrapping around the circular genome.
    """
    seq_len = len(sequence)
    # Convert to 0-indexed
    center = pos - 1
    
    indices = []
    for offset in range(-window_size, window_size + 1):
        idx = (center + offset) % seq_len
        indices.append(idx)
    
    return ''.join(sequence[i] for i in indices)


def compute_kmer_vector(sequence, k=4):
    """
    Compute k-mer frequency vector for a sequence.
    Returns a numpy array of length 4^k (256 for 4-mers).
    """
    # Generate all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    
    # Count k-mers in sequence
    counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_to_idx:  # Skip if contains N
            counts[kmer] += 1
    
    # Convert to frequency vector
    vector = np.zeros(len(all_kmers))
    total = sum(counts.values())
    if total > 0:
        for kmer, count in counts.items():
            vector[kmer_to_idx[kmer]] = count / total
    
    return vector


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    # Load sequence
    fasta_path = os.path.join(project_dir, 'data', 'sequence.fasta')
    sequence = load_fasta_sequence(fasta_path)
    print(f"Loaded genome sequence: {len(sequence)} bp")
    
    # Load variants
    variants = pd.read_csv(os.path.join(intermediate_dir, 'merged_variants.csv'))
    print(f"Loaded {len(variants)} variants")
    
    # Load existing graph
    pickle_path = os.path.join(intermediate_dir, 'mitograph_base.pkl')
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compute k-mer vectors for all variant positions
    print(f"\nComputing 4-mer vectors (±20bp window)...")
    k = 4
    window_size = 20
    
    positions = variants['pos'].unique()
    pos_to_vector = {}
    
    for pos in positions:
        window = get_circular_window(sequence, pos, window_size)
        vec = compute_kmer_vector(window, k)
        pos_to_vector[pos] = vec
    
    print(f"Computed vectors for {len(pos_to_vector)} unique positions")
    
    # Compute pairwise cosine similarity and add edges
    # To avoid O(n^2) for all pairs, only compare variants within a reasonable
    # positional distance OR use a threshold approach
    print(f"\nComputing pairwise k-mer similarities...")
    similarity_threshold = 0.85
    
    kmer_edges = []
    positions_list = sorted(pos_to_vector.keys())
    n = len(positions_list)
    
    # Create variant ID lookup
    var_ids_by_pos = {}
    for _, var in variants.iterrows():
        var_id = f"var_{int(var['pos'])}_{var['ref']}_{var['alt']}"
        pos = int(var['pos'])
        if pos not in var_ids_by_pos:
            var_ids_by_pos[pos] = []
        var_ids_by_pos[pos].append(var_id)
    
    # Compare all pairs efficiently
    # Since vectors are 256-dim, we can use matrix operations
    all_vectors = np.array([pos_to_vector[p] for p in positions_list])
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = all_vectors / norms
    
    # Compute similarity matrix in chunks to manage memory
    chunk_size = 500
    edge_count = 0
    
    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        sim_chunk = normalized[i:chunk_end] @ normalized.T
        
        for row_idx in range(chunk_end - i):
            global_i = i + row_idx
            for j in range(global_i + 1, n):
                sim = sim_chunk[row_idx, j]
                if sim >= similarity_threshold:
                    pos_i = positions_list[global_i]
                    pos_j = positions_list[j]
                    
                    # Add edges between all variant pairs at these positions
                    for var_id_i in var_ids_by_pos.get(pos_i, []):
                        for var_id_j in var_ids_by_pos.get(pos_j, []):
                            if var_id_i != var_id_j:
                                G.add_edge(var_id_i, var_id_j, 
                                          edge_type='KMER_SIMILARITY',
                                          score=float(sim))
                                kmer_edges.append({
                                    'variant_1': var_id_i,
                                    'variant_2': var_id_j,
                                    'position_1': pos_i,
                                    'position_2': pos_j,
                                    'similarity': float(sim)
                                })
                                edge_count += 1
        
        if (i // chunk_size) % 2 == 0:
            print(f"  Processed {chunk_end}/{n} positions, {edge_count} edges so far")
    
    print(f"\nTotal k-mer similarity edges: {edge_count}")
    
    # Save k-mer edges
    kmer_df = pd.DataFrame(kmer_edges)
    kmer_path = os.path.join(intermediate_dir, 'kmer_edges.csv')
    kmer_df.to_csv(kmer_path, index=False)
    print(f"Saved k-mer edges to {kmer_path}")
    
    # Save updated graph
    pickle_path_updated = os.path.join(intermediate_dir, 'mitograph_full.pkl')
    with open(pickle_path_updated, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved updated graph to {pickle_path_updated}")
    
    # Updated stats
    print(f"\n--- Updated Graph Statistics ---")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    
    edge_types = {}
    for _, _, data in G.edges(data=True):
        et = data.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1
    print(f"Edge types: {edge_types}")
    
    if len(kmer_edges) > 0:
        sims = [e['similarity'] for e in kmer_edges]
        print(f"\nK-mer similarity stats:")
        print(f"  Min: {min(sims):.4f}")
        print(f"  Max: {max(sims):.4f}")
        print(f"  Mean: {np.mean(sims):.4f}")

if __name__ == '__main__':
    main()
