# MitoGraph: Mitochondrial Knowledge Graph for VUS Pathogenicity Prediction

A Graph ML pipeline that builds a heterogeneous Knowledge Graph from mitochondrial variant databases and uses link prediction to assess Variants of Uncertain Significance (VUS).

**🌐 [Live Dashboard →](https://mitomap-app.vercel.app/)**

## Overview

MitoGraph integrates three data sources - **RefSeq GFF3** (gene annotations), **ClinVar** (variant classifications), and **MITOMAP** (disease associations, conservation scores) - into a single Knowledge Graph. A Graph Neural Network (GATv2Conv-based heterogeneous encoder with attention) is trained on known pathogenic variant-phenotype associations, then used to predict potential disease links for VUS.

### Key Results
- **Test AUPRC: 0.792** | **Test AUROC: 0.765** | **Silhouette: 0.577**
- 1,228 VUS scored against 808 disease phenotypes
- 482 VUS flagged as potentially pathogenic (39%)

## Interactive Dashboard

The results are served through an interactive Next.js dashboard deployed on Vercel.

**Stats overview + UMAP latent space** - 2D projection of GATv2Conv variant embeddings colored by pathogenicity class. Flagged VUS (★) cluster with known pathogenic variants.

![Dashboard overview showing stat cards and UMAP scatter plot](docs/dashboard_overview.png)

**Mitochondrial complex graph** - Force-directed layout of the knowledge graph hierarchy: Complexes → Genes → Variants → Phenotypes. Nodes are draggable, clickable, and color-coded by type.

![Interactive network graph with gene labels and complex hierarchy](docs/network_graph.png)

**Dashboard source code:** [Shreyan-A0I/Mitomap-app](https://github.com/Shreyan-A0I/Mitomap-app)

## Graph Structure

| Node Type | Count | Features |
|-----------|-------|----------|
| Variant | 3,439 | PhyloP conservation, clinical significance, circular positional encoding, APOGEE/MitoTIP scores |
| Gene | 37 | Biotype (tRNA, rRNA, protein-coding) |
| Complex | 4 | Respiratory chain complex (I, III, IV, V) |
| Phenotype | 808 | Disease names from ClinVar + MITOMAP |

| Edge Type | Count | Description |
|-----------|-------|-------------|
| LOCATED_IN | 3,429 | Variant → Gene (positional overlap) |
| PART_OF | 13 | Gene → Complex (respiratory chain mapping) |
| ASSOCIATED_WITH | 620 | Variant → Phenotype (confirmed associations) |
| KMER_SIMILARITY | 7,437 | Variant ↔ Variant (4-mer cosine sim > 0.85, ±20bp circular window) |

## Pipeline

```
src/etl/                  src/graph/                src/ml/
┌───────────────────┐    ┌──────────────────┐      ┌──────────────────┐
│ parse_gff3.py     │    │ build_graph.py   │      │ graph_to_pyg.py  │
│ parse_clinvar.py  │───▶│ kmer_similarity  │─────▶│ model.py (GAT)   │
│ extract_mitomap.py│    │ export_graph.py  │      │ train.py         │
│ merge_variants.py │    └──────────────────┘      │ predict_vus.py   │
│ build_complex_map │                              │ export_graph_json │
└───────────────────┘                              └──────────────────┘
```

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mitograph

# Install ML dependencies
conda run -n mitograph python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n mitograph python -m pip install torch-geometric umap-learn
```

## Design Decisions

- **GATv2Conv + 4 Attention Heads**: Dynamic attention learns which neighbor edges matter most for pathogenicity prediction
- **Hard Negative Mining**: 1,611 benign variants forced to score 0.0 against all phenotypes, reduced VUS false-positive rate from 74% to 39%
- **Circular Positional Encoding**: mtDNA is circular; positions are encoded as `(sin(2π·pos/16569), cos(2π·pos/16569))` so position 16569 neighbors position 1
- **PhyloP Conservation**: 100-vertebrate basewise PhyloP scores from UCSC; missing values imputed with median (no 0.0 placeholders)
- **4-mer Similarity**: ±20bp windows on the circular genome; cosine similarity threshold of 0.85
- **Variant-Level Split**: Entire variants held out for val/test to prevent edge leakage through k-mer similarity edges
- **DBSCAN Clustering**: eps=0.4, min_samples=5 on UMAP embeddings to identify pathogenic clusters (Silhouette=0.577)
