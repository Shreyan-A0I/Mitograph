# MitoGraph: Mitochondrial Knowledge Graph for VUS Pathogenicity Prediction

A Graph ML pipeline that builds a heterogeneous Knowledge Graph from mitochondrial variant databases and uses link prediction to assess Variants of Uncertain Significance (VUS).

## Overview

MitoGraph integrates three data sources — **RefSeq GFF3** (gene annotations), **ClinVar** (variant classifications), and **MITOMAP** (disease associations, conservation scores) — into a single Knowledge Graph. A Graph Neural Network (GATv2Conv-based heterogeneous encoder with attention) is trained on known pathogenic variant–phenotype associations, then used to predict potential disease links for VUS.

### Key Results
- **Test AUPRC: 0.80** | **Test AUROC: 0.77**
- 1,228 VUS scored against 808 disease phenotypes

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
│ build_complex_map │                              └──────────────────┘
└───────────────────┘
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
# ETL (data must be in data/)
python src/etl/parse_gff3.py
python src/etl/parse_clinvar.py
python src/etl/extract_mitomap.py
python src/etl/merge_variants.py
python src/etl/build_complex_mapping.py

# Knowledge Graph
python src/graph/build_graph.py
python src/graph/kmer_similarity.py
python src/graph/export_graph.py

# Graph ML
python src/ml/graph_to_pyg.py
python src/ml/train.py
python src/ml/predict_vus.py
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
