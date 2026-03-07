# MitoGraph: Mitochondrial Knowledge Graph for VUS Pathogenicity Prediction

A Graph ML pipeline that builds a heterogeneous Knowledge Graph from mitochondrial variant databases and uses link prediction to assess Variants of Uncertain Significance (VUS).

## Overview

MitoGraph integrates three data sources — **RefSeq GFF3** (gene annotations), **ClinVar** (variant classifications), and **MITOMAP** (disease associations, conservation scores) — into a single Knowledge Graph. A Graph Neural Network (SAGEConv-based heterogeneous encoder) is trained on known pathogenic variant–phenotype associations, then used to predict potential disease links for VUS.

### Key Results
- **Test AUPRC: 0.80** | **Test AUROC: 0.77**
- 1,228 VUS scored against 808 disease phenotypes
- 911 VUS flagged as clustering with pathogenic variants in latent space

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
Phase A: ETL                    Phase B: Graph              Phase C: ML
┌─────────────────┐            ┌──────────────────┐        ┌──────────────────┐
│ a1: Parse GFF3  │            │ b1: Build graph  │        │ c1: → PyG        │
│ a2: Parse ClinVar│───────────│ b2: K-mer edges  │────────│ c2: RGCN model   │
│ a3: MITOMAP SQL │            │ b3: Export stats │        │ c3: Train        │
│ a4: Merge + PhyloP│          └──────────────────┘        │ c4: Predict VUS  │
│ a5: Complex map │                                        └──────────────────┘
└─────────────────┘
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

## Usage

```bash
# Phase A: ETL (data must be in data/)
python src/a1_parse_gff3.py
python src/a2_parse_clinvar.py
python src/a3_extract_mitomap.py
python src/a4_merge_variants.py
python src/a5_build_complex_mapping.py

# Phase B: Knowledge Graph
python src/b1_build_graph.py
python src/b2_kmer_similarity.py
python src/b3_export_graph.py

# Phase C: Graph ML
python src/c1_graph_to_pyg.py
python src/c3_train.py
python src/c4_predict_vus.py
```

## Design Decisions

- **Circular Positional Encoding**: mtDNA is circular; positions are encoded as `(sin(2π·pos/16569), cos(2π·pos/16569))` so position 16569 neighbors position 1
- **PhyloP Conservation**: 100-vertebrate basewise PhyloP scores from UCSC; missing values imputed with median (no 0.0 placeholders)
- **4-mer Similarity**: ±20bp windows on the circular genome; cosine similarity threshold of 0.85
- **Variant-Level Split**: Entire variants held out for val/test to prevent edge leakage through k-mer similarity edges

## Data

Raw data files are excluded from version control (see `.gitignore`). Required files in `data/`:
- `sequence.gff3` — RefSeq gene annotations
- `clinvar_mt_variant_summary.csv` — ClinVar mitochondrial variants
- `mitomap.dump.sql` — MITOMAP PostgreSQL dump
- `sequence.fasta` — rCRS mitochondrial genome
- `conservation_scores.txt` — PhyloP basewise scores
