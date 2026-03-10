# MitoGraph: Mitochondrial Knowledge Graph for VUS Pathogenicity Prediction

A Graph ML pipeline that builds a heterogeneous Knowledge Graph from mitochondrial variant databases and uses link prediction to assess Variants of Uncertain Significance (VUS).

**🌐 [Live Dashboard →](https://mitomap-app.vercel.app/)**

## Overview

MitoGraph integrates three data sources - **RefSeq GFF3** (gene annotations), **ClinVar** (variant classifications), and **MITOMAP** (disease associations, conservation scores) - into a single Knowledge Graph. A Graph Neural Network (GATv2Conv-based heterogeneous encoder with attention) is trained on known pathogenic variant-phenotype associations, then used to predict potential disease links for VUS.

### Key Results
- **Test AUPRC: 0.830** | **Test AUROC: 0.789** | **Silhouette: 0.582**
- 1,228 VUS scored against 808 disease phenotypes
- Model config: GATv2Conv with 64 dimensions, 8 attention heads, LR 0.005

## Interactive Dashboard

The results are served through an interactive Next.js dashboard deployed on Vercel.

**Feature Importance (Biological Grounding)** - Extracted linear projection weights from the model's input encoder prove that MitoGraph focuses on biologically relevant features (PhyloP, mutation type) without relying on data leakage.

![Feature Importance Model Diagnostics](docs/feature_importance.png)

**Network Graph (Structural Interpretability)** - Force-directed layout of the knowledge graph hierarchy: Complexes → Genes → Variants → Phenotypes. Nodes are draggable, clickable, and edge thickness represents GATv2Conv attention weight ($\alpha$).

![Interactive network graph with gene labels and complex hierarchy](docs/network_graph.png)

**Stats overview + UMAP latent space** - 2D projection of GATv2Conv variant embeddings colored by pathogenicity class. Flagged VUS (★) cluster with pathogenic variants. Extracted explicitly from the Graph structure's deepest layer.

![Dashboard overview showing stat cards and UMAP scatter plot](docs/dashboard_overview.png)

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

### Node Feature Vectors

Each node type has a fixed-length feature vector fed to the GATv2Conv encoder:

**Variant (14D):**

| Dim | Feature | Description |
|-----|---------|-------------|
| 0 | PhyloP conservation | 100-vertebrate basewise score (median-imputed) |
| 1-2 | Circular position | Encoded as `sin(2π·pos/16569)` and `cos(2π·pos/16569)` |
| 3 | is_transition | 1 if A↔G or C↔T (biochemically more tolerated) |
| 4 | is_transversion | 1 if A↔C, A↔T, C↔G, G↔T (structurally disruptive) |
| 5 | is_indel | 1 if insertion or deletion |
| 6-9 | Reference Nucleotide | One-hot encoded `[A, C, G, T]` (all 0 if indel) |
| 10-13 | Alternate Nucleotide | One-hot encoded `[A, C, G, T]` (all 0 if indel) |

**Gene (3D):** one-hot `[tRNA, rRNA, protein_coding]`

**Complex (4D):** one-hot `[I, III, IV, V]`

**Phenotype (384D):** Generated offline using `sentence-transformers` (`all-MiniLM-L6-v2`) on disease names, allowing the model to naturally understand that "Optic Atrophy" and "Blindness" are related concepts before message passing even begins.

## Design Decisions

- **GATv2Conv + 8 Attention Heads**: Dynamic attention learns which neighbor edges matter most for pathogenicity prediction
- **Strict Data Integrity**: Removed all clinical significance flags (is_pathogenic) and previously computed ML scores (APOGEE) from the Variant feature vector to completely eliminate data leakage.
- **Hard Negative Mining**: Benign variants forced to score 0.0 against all phenotypes to teach the model what "healthy" graph topology looks like.
- **LLM Embeddings (all-MiniLM-L6-v2)**: Replaced random JL projection with semantic text embeddings for phenotype names.
- **Circular Positional Encoding**: mtDNA is circular; positions are encoded as `(sin(2π·pos/16569), cos(2π·pos/16569))` so position 16569 neighbors position 1
- **PhyloP Conservation**: 100-vertebrate basewise PhyloP scores from UCSC; missing values imputed with median (no 0.0 placeholders)
- **Variant-Level Split**: Entire variants held out for val/test to prevent edge leakage through k-mer similarity edges
- **DBSCAN Clustering**: eps=0.4, min_samples=5 on UMAP embeddings to identify pathogenic clusters (Silhouette=0.582)

## Data Sources & Acknowledgements

MitoGraph's neuro-symbolic architecture relies on the expert curation and open-access data provided by the following institutions:

- **MITOMAP**: The definitive human mitochondrial genome database. MITOMAP provided the expertly curated variant-to-phenotype linkages, functional classifications (mmut, rtmut), and the machine-learning-derived APOGEE pathogenicity probabilities.
- **ClinVar (NCBI)**: A freely accessible, public archive of reports of the relationships among human variations and phenotypes. ClinVar provided the foundational baseline of clinical observations and the primary set of Variants of Uncertain Significance (VUS) for this project's predictive modeling.
- **rCRS (Revised Cambridge Reference Sequence)**: Sourced via NCBI RefSeq (NC_012920.1), this 16,569 bp sequence served as the physical coordinate system for the knowledge graph and k-mer similarity generation.
- **PhyloP (UCSC Genome Browser)**: Provided the base-by-base evolutionary conservation scores used as primary structural features for the Graph Attention Network.
