#!/usr/bin/env python3
"""
Training pipeline with variant-level split and hard negative mining.

Implements:
  - Variant-level train/val/test split (70/15/15) to prevent edge leakage
  - Hard negative mining: benign variants forced to score 0.0 against all phenotypes
  - Random negative sampling for standard BCE loss
  - Early stopping on validation AUPRC

The key insight: we hold out ENTIRE VARIANTS for val/test, so the model
must predict phenotype associations for completely unseen variants.

Hard negative mining teaches the model what "benign" looks like — without it,
the model has no signal to suppress scores for variants with low conservation
and benign graph topology, leading to ~74% false-positive rate on VUS.

Output: data/results/training_metrics.json
        data/results/model_checkpoint.pt
"""

import os
import sys
import json
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve
)

from model import MitoGraphLinkPredictor


def variant_level_split(hetero_data, metadata, train_ratio=0.70,
                        val_ratio=0.15, seed=42):
    """
    Split ASSOCIATED_WITH edges by holding out entire variants.

    Instead of randomly splitting edges (which causes leakage when a test
    variant has KMER_SIMILARITY edges to training variants through the
    ASSOCIATED_WITH target), we hold out entire variants so the model must
    generalize to completely unseen VUS-like scenarios.

    Returns:
        train_edge_index, val_edge_index, test_edge_index
        Each is [2, num_edges] tensor
    """
    edge_key = ('variant', 'associated_with', 'phenotype')
    edge_index = hetero_data[edge_key].edge_index

    var_labels = metadata['var_labels']

    # Get unique source variants in the ASSOCIATED_WITH edges
    src_variants = edge_index[0].unique().tolist()

    # Separate pathogenic/likely_pathogenic variants for stratified split
    # (we want both train and test to have pathogenic examples)
    path_variants = []
    other_variants = []
    for v_idx in src_variants:
        label = var_labels.get(str(v_idx), '').lower()
        if 'pathogenic' in label:
            path_variants.append(v_idx)
        else:
            other_variants.append(v_idx)

    random.seed(seed)
    random.shuffle(path_variants)
    random.shuffle(other_variants)

    def split_list(lst, r1, r2):
        n = len(lst)
        t1 = int(n * r1)
        t2 = int(n * (r1 + r2))
        return lst[:t1], lst[t1:t2], lst[t2:]

    path_train, path_val, path_test = split_list(
        path_variants, train_ratio, val_ratio)
    other_train, other_val, other_test = split_list(
        other_variants, train_ratio, val_ratio)

    train_vars = set(path_train + other_train)
    val_vars = set(path_val + other_val)
    test_vars = set(path_test + other_test)

    # Split edges based on variant membership
    train_mask = torch.tensor(
        [edge_index[0, i].item() in train_vars for i in range(edge_index.shape[1])]
    )
    val_mask = torch.tensor(
        [edge_index[0, i].item() in val_vars for i in range(edge_index.shape[1])]
    )
    test_mask = torch.tensor(
        [edge_index[0, i].item() in test_vars for i in range(edge_index.shape[1])]
    )

    train_ei = edge_index[:, train_mask]
    val_ei = edge_index[:, val_mask]
    test_ei = edge_index[:, test_mask]

    print(f"Variant-level split:")
    print(f"  Train: {len(train_vars)} variants, {train_ei.shape[1]} edges")
    print(f"  Val:   {len(val_vars)} variants, {val_ei.shape[1]} edges")
    print(f"  Test:  {len(test_vars)} variants, {test_ei.shape[1]} edges")

    return train_ei, val_ei, test_ei


def negative_sampling(pos_edge_index, num_variants, num_phenotypes, num_neg=1):
    """
    Generate random negative (variant, phenotype) pairs.
    For each positive edge, sample `num_neg` random phenotypes not in the
    positive set for that variant.
    """
    pos_set = set()
    for i in range(pos_edge_index.shape[1]):
        v = pos_edge_index[0, i].item()
        p = pos_edge_index[1, i].item()
        pos_set.add((v, p))

    neg_src, neg_dst = [], []
    for i in range(pos_edge_index.shape[1]):
        v = pos_edge_index[0, i].item()
        for _ in range(num_neg):
            # Sample random phenotype not in positive set
            while True:
                p = random.randint(0, num_phenotypes - 1)
                if (v, p) not in pos_set:
                    break
            neg_src.append(v)
            neg_dst.append(p)

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def hard_negative_sampling(var_labels, num_phenotypes, max_samples_per_epoch=500):
    """
    Hard negative mining: sample (benign_variant, phenotype) pairs.

    The model must learn to predict score=0.0 for these pairs, teaching it
    that variants with benign conservation profiles and benign graph topology
    should NOT be linked to any phenotype.

    This directly addresses the false-positive problem where the model flags
    74% of VUS as pathogenic because it never learned what "benign" looks like.

    Args:
        var_labels: dict mapping variant_idx -> clinical significance label
        num_phenotypes: total number of phenotype nodes
        max_samples_per_epoch: cap per epoch to avoid overwhelming the loss

    Returns:
        hard_neg_edge_index: [2, num_hard_neg] tensor
    """
    # Collect all benign/likely benign variant indices
    benign_indices = []
    for idx_str, label in var_labels.items():
        label_lower = label.lower()
        if 'benign' in label_lower:
            benign_indices.append(int(idx_str))

    if not benign_indices:
        return torch.zeros((2, 0), dtype=torch.long)

    # Sample benign variants and random phenotypes
    hard_src, hard_dst = [], []
    n_per_variant = max(1, max_samples_per_epoch // len(benign_indices))

    for v_idx in benign_indices:
        # Sample a few random phenotypes for each benign variant
        sampled_phenos = random.sample(
            range(num_phenotypes), min(n_per_variant, num_phenotypes)
        )
        for p_idx in sampled_phenos:
            hard_src.append(v_idx)
            hard_dst.append(p_idx)

    # Subsample if too many
    if len(hard_src) > max_samples_per_epoch:
        indices = random.sample(range(len(hard_src)), max_samples_per_epoch)
        hard_src = [hard_src[i] for i in indices]
        hard_dst = [hard_dst[i] for i in indices]

    return torch.tensor([hard_src, hard_dst], dtype=torch.long)


def build_message_passing_edges(hetero_data, train_edge_index):
    """
    Build the edge_index_dict for message passing during training.
    Uses ALL structural edges (LOCATED_IN, PART_OF, KMER_SIMILARITY)
    but ONLY training ASSOCIATED_WITH edges.
    """
    edge_index_dict = {}

    for edge_type in hetero_data.edge_types:
        if edge_type == ('variant', 'associated_with', 'phenotype'):
            edge_index_dict[edge_type] = train_edge_index
        elif edge_type == ('phenotype', 'rev_associated_with', 'variant'):
            # Reverse of training edges only
            edge_index_dict[edge_type] = torch.stack(
                [train_edge_index[1], train_edge_index[0]])
        else:
            edge_index_dict[edge_type] = hetero_data[edge_type].edge_index

    return edge_index_dict


def compute_metrics(model, z_dict, edge_index, neg_edge_index):
    """Compute AUPRC and AUROC for a set of pos/neg edges."""
    with torch.no_grad():
        pos_scores = model.predict_links(z_dict, edge_index).sigmoid()
        neg_scores = model.predict_links(z_dict, neg_edge_index).sigmoid()

        y_true = torch.cat([
            torch.ones(pos_scores.shape[0]),
            torch.zeros(neg_scores.shape[0])
        ]).numpy()
        y_scores = torch.cat([pos_scores, neg_scores]).numpy()

        auprc = average_precision_score(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)

    return auprc, auroc


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    results_dir = os.path.join(project_dir, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- Load data ---
    data_path = os.path.join(intermediate_dir, 'hetero_data.pt')
    hetero_data = torch.load(data_path, weights_only=False)
    print(f"Loaded HeteroData: {hetero_data}")

    meta_path = os.path.join(intermediate_dir, 'graph_metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # --- Split edges ---
    train_ei, val_ei, test_ei = variant_level_split(hetero_data, metadata)

    num_variants = hetero_data['variant'].x.shape[0]
    num_phenotypes = hetero_data['phenotype'].x.shape[0]
    pheno_in_dim = hetero_data['phenotype'].x.shape[1]

    # Count benign variants for hard negative mining
    benign_count = sum(1 for l in metadata['var_labels'].values()
                       if 'benign' in l.lower())
    print(f"\nBenign/Likely benign variants available for hard negatives: {benign_count}")

    # --- Initialize model ---
    mp_edge_index_dict = build_message_passing_edges(hetero_data, train_ei)
    model_metadata = hetero_data.metadata()

    model = MitoGraphLinkPredictor(
        metadata=model_metadata,
        hidden_dim=args.hidden_dim,
        out_dim=32,
        variant_in_dim=hetero_data['variant'].x.shape[1],
        gene_in_dim=hetero_data['gene'].x.shape[1],
        complex_in_dim=hetero_data['complex'].x.shape[1],
        phenotype_in_dim=pheno_in_dim,
        heads=args.heads,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # --- Training ---
    n_epochs = 200
    patience = 30
    best_val_auprc = 0.0
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {n_epochs} epochs (patience={patience})...")
    print(f"  Using: GATv2Conv ({args.heads} heads, dim={args.hidden_dim}, lr={args.lr}) + Hard Negative Mining")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Train AUPRC':>11} | {'Val AUPRC':>10} | {'Val AUROC':>10}")
    print('-' * 60)

    for epoch in range(1, n_epochs + 1):
        model.train()

        # Generate negative samples for this epoch
        neg_train_ei = negative_sampling(
            train_ei, num_variants, num_phenotypes, num_neg=3)

        # Hard negative mining: benign variants × random phenotypes → score 0.0
        hard_neg_ei = hard_negative_sampling(
            metadata['var_labels'], num_phenotypes, max_samples_per_epoch=500)

        # Forward pass
        x_dict = {nt: hetero_data[nt].x for nt in hetero_data.node_types}
        z_dict = model(x_dict, mp_edge_index_dict)

        # Positive edge scores (pathogenic → phenotype)
        pos_scores = model.predict_links(z_dict, train_ei)
        # Random negative edge scores
        neg_scores = model.predict_links(z_dict, neg_train_ei)

        # Loss: BCE for pos + neg
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores))

        # Hard negative loss: force benign variants to score 0.0 against phenotypes
        if hard_neg_ei.shape[1] > 0:
            hard_neg_scores = model.predict_links(z_dict, hard_neg_ei)
            hard_neg_loss = F.binary_cross_entropy_with_logits(
                hard_neg_scores, torch.zeros_like(hard_neg_scores))
        else:
            hard_neg_loss = torch.tensor(0.0)

        loss = pos_loss + neg_loss + hard_neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Evaluate ---
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_dict_eval = model(x_dict, mp_edge_index_dict)

            # Train metrics
            neg_train_eval = negative_sampling(
                train_ei, num_variants, num_phenotypes, num_neg=1)
            train_auprc, _ = compute_metrics(
                model, z_dict_eval, train_ei, neg_train_eval)

            # Val metrics
            neg_val_ei = negative_sampling(
                val_ei, num_variants, num_phenotypes, num_neg=1)
            val_auprc, val_auroc = compute_metrics(
                model, z_dict_eval, val_ei, neg_val_ei)

            print(f"{epoch:>5} | {loss.item():>8.4f} | "
                  f"{train_auprc:>11.4f} | {val_auprc:>10.4f} | {val_auroc:>10.4f}")

            history.append({
                'epoch': epoch,
                'loss': float(loss.item()),
                'train_auprc': float(train_auprc),
                'val_auprc': float(val_auprc),
                'val_auroc': float(val_auroc),
            })

            # Early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                # Save best model
                checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_auprc': val_auprc,
                    'val_auroc': val_auroc,
                    'epoch': epoch,
                }, checkpoint_path)
            else:
                patience_counter += 5  # since we evaluate every 5 epochs

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val AUPRC: {best_val_auprc:.4f})")
                break

    # --- Final test evaluation ---
    print(f"\n--- Final Test Evaluation ---")
    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        x_dict = {nt: hetero_data[nt].x for nt in hetero_data.node_types}
        z_dict_test = model(x_dict, mp_edge_index_dict)

    neg_test_ei = negative_sampling(
        test_ei, num_variants, num_phenotypes, num_neg=1)
    test_auprc, test_auroc = compute_metrics(
        model, z_dict_test, test_ei, neg_test_ei)

    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Best model from epoch: {checkpoint['epoch']}")

    # --- Save final metrics ---
    final_metrics = {
        'best_epoch': checkpoint['epoch'],
        'best_val_auprc': float(best_val_auprc),
        'test_auprc': float(test_auprc),
        'test_auroc': float(test_auroc),
        'training_history': history,
        'split_info': {
            'train_edges': int(train_ei.shape[1]),
            'val_edges': int(val_ei.shape[1]),
            'test_edges': int(test_ei.shape[1]),
        },
        'model_config': {
            'architecture': 'GATv2Conv',
            'attention_heads': args.heads,
            'hidden_dim': args.hidden_dim,
            'lr': args.lr,
            'out_dim': 32,
            'hard_negative_mining': True,
        }
    }

    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save final embeddings for VUS prediction
    embeddings_path = os.path.join(intermediate_dir, 'node_embeddings.pt')
    torch.save({k: v.detach() for k, v in z_dict_test.items()}, embeddings_path)
    print(f"Saved embeddings to {embeddings_path}")


if __name__ == '__main__':
    main()
