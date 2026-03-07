#!/usr/bin/env python3
"""
Phase C - Step 4: Score VUS variants and visualize latent space.

For each VUS variant:
  1. Compute predicted link scores to all phenotypes
  2. Flag VUS that cluster with Pathogenic variants in latent space
  3. Generate UMAP + DBSCAN clustering visualization

Output: data/results/vus_predictions.csv
        data/results/latent_space.png
        data/results/vus_summary.json
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Using PCA for dimensionality reduction.")
    from sklearn.decomposition import PCA


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    results_dir = os.path.join(project_dir, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- Load embeddings and metadata ---
    embeddings_path = os.path.join(intermediate_dir, 'node_embeddings.pt')
    embeddings = torch.load(embeddings_path, weights_only=False)
    print(f"Loaded embeddings: {', '.join(f'{k}: {v.shape}' for k, v in embeddings.items())}")

    meta_path = os.path.join(intermediate_dir, 'graph_metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    var_labels = metadata['var_labels']
    idx_to_var = metadata['idx_to_var']
    idx_to_pheno = metadata.get('idx_to_pheno', {})
    pheno_names = metadata.get('pheno_names', {})

    variant_emb = embeddings['variant'].numpy()
    phenotype_emb = embeddings['phenotype'].numpy()

    n_variants = variant_emb.shape[0]
    n_phenotypes = phenotype_emb.shape[0]

    print(f"Variants: {n_variants}, Phenotypes: {n_phenotypes}")

    # --- Identify VUS variants ---
    vus_indices = []
    path_indices = []
    benign_indices = []
    other_indices = []

    for idx in range(n_variants):
        label = var_labels.get(str(idx), '').lower()
        if 'uncertain' in label or 'vus' in label:
            vus_indices.append(idx)
        elif 'pathogenic' in label:
            path_indices.append(idx)
        elif 'benign' in label:
            benign_indices.append(idx)
        else:
            other_indices.append(idx)

    print(f"\nVariant classification:")
    print(f"  Pathogenic/Likely: {len(path_indices)}")
    print(f"  Benign/Likely:     {len(benign_indices)}")
    print(f"  VUS:               {len(vus_indices)}")
    print(f"  Other:             {len(other_indices)}")

    # --- Score VUS → Phenotype links ---
    print(f"\nScoring VUS variants against {n_phenotypes} phenotypes...")

    predictions = []
    for vus_idx in vus_indices:
        var_id = idx_to_var.get(str(vus_idx), f'unknown_{vus_idx}')
        vus_emb = variant_emb[vus_idx]

        # Dot-product score against ALL phenotypes
        scores = phenotype_emb @ vus_emb  # [n_phenotypes]
        scores_clipped = np.clip(scores, -50, 50)
        probs = 1 / (1 + np.exp(-scores_clipped))  # sigmoid

        # Get top-K predictions
        top_k = min(5, n_phenotypes)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for rank, pheno_idx in enumerate(top_indices):
            pheno_name = pheno_names.get(str(pheno_idx), f'Phenotype_{pheno_idx}')
            predictions.append({
                'variant_id': var_id,
                'variant_idx': int(vus_idx),
                'phenotype': pheno_name,
                'phenotype_idx': int(pheno_idx),
                'score': float(scores[pheno_idx]),
                'probability': float(probs[pheno_idx]),
                'rank': rank + 1,
            })

    pred_df = pd.DataFrame(predictions)
    pred_path = os.path.join(results_dir, 'vus_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved {len(predictions)} VUS predictions to {pred_path}")

    # Show top VUS predictions
    if len(predictions) > 0:
        top_preds = pred_df[pred_df['rank'] == 1].nlargest(20, 'probability')
        print(f"\nTop 20 VUS → Phenotype predictions (rank 1):")
        for _, row in top_preds.iterrows():
            print(f"  {row['variant_id']:>25s} → {row['phenotype']:<40s} "
                  f"(p={row['probability']:.4f})")

    # --- DBSCAN clustering in embedding space ---
    # Identify VUS that cluster with pathogenic variants
    print(f"\nClustering variants in embedding space...")

    all_indices = path_indices + benign_indices + vus_indices
    all_embs = variant_emb[all_indices]

    scaler = StandardScaler()
    all_embs_scaled = scaler.fit_transform(all_embs)

    db = DBSCAN(eps=0.8, min_samples=3)
    cluster_labels = db.fit_predict(all_embs_scaled)

    # Assign labels to each group
    n_path = len(path_indices)
    n_benign = len(benign_indices)
    n_vus = len(vus_indices)

    path_clusters = set(cluster_labels[:n_path])
    path_clusters.discard(-1)  # Remove noise label

    benign_clusters = set(cluster_labels[n_path:n_path + n_benign])
    benign_clusters.discard(-1)

    vus_cluster_labels = cluster_labels[n_path + n_benign:]

    # Flag VUS in pathogenic clusters
    vus_in_path_clusters = []
    for i, vus_idx in enumerate(vus_indices):
        cl = vus_cluster_labels[i]
        if cl in path_clusters and cl != -1:
            var_id = idx_to_var.get(str(vus_idx), f'unknown_{vus_idx}')
            vus_in_path_clusters.append({
                'variant_id': var_id,
                'variant_idx': int(vus_idx),
                'cluster': int(cl),
                'in_pathogenic_cluster': True,
            })

    print(f"VUS in pathogenic clusters: {len(vus_in_path_clusters)}/{n_vus}")

    # --- UMAP / PCA visualization ---
    print(f"\nGenerating latent space visualization...")

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                            min_dist=0.1, metric='cosine')
        coords = reducer.fit_transform(all_embs_scaled)
    else:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(all_embs_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each category
    ax.scatter(coords[:n_path, 0], coords[:n_path, 1],
               c='#e74c3c', s=30, label=f'Pathogenic/LP ({n_path})',
               alpha=0.8, edgecolors='darkred', linewidths=0.5)
    ax.scatter(coords[n_path:n_path + n_benign, 0],
               coords[n_path:n_path + n_benign, 1],
               c='#2ecc71', s=30, label=f'Benign/LB ({n_benign})',
               alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    ax.scatter(coords[n_path + n_benign:, 0], coords[n_path + n_benign:, 1],
               c='#3498db', s=20, label=f'VUS ({n_vus})',
               alpha=0.5, edgecolors='navy', linewidths=0.3)

    # Highlight VUS in pathogenic clusters
    if vus_in_path_clusters:
        flagged_local_indices = [
            n_path + n_benign + vus_indices.index(v['variant_idx'])
            for v in vus_in_path_clusters
        ]
        ax.scatter(coords[flagged_local_indices, 0],
                   coords[flagged_local_indices, 1],
                   c='#f39c12', s=80, marker='*',
                   label=f'VUS in Path. cluster ({len(vus_in_path_clusters)})',
                   alpha=0.9, edgecolors='darkorange', linewidths=0.5, zorder=5)

    method = 'UMAP' if HAS_UMAP else 'PCA'
    ax.set_title(f'MitoGraph: Variant Latent Space ({method})', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(results_dir, 'latent_space.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {fig_path}")

    # --- Summary ---
    summary = {
        'n_vus': n_vus,
        'n_pathogenic': n_path,
        'n_benign': n_benign,
        'vus_in_pathogenic_clusters': len(vus_in_path_clusters),
        'vus_flagged': vus_in_path_clusters,
        'total_predictions': len(predictions),
        'visualization_method': method,
    }

    summary_path = os.path.join(results_dir, 'vus_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
