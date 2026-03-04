#!/usr/bin/env python3
"""
Phase A - Step 1: Parse GFF3 file to extract gene annotations.
Reads sequence.gff3 (RefSeq NC_012920.1) and outputs a clean CSV of genes
with their names, coordinates, and biotypes.

Output: data/intermediate/genes.csv
"""

import pandas as pd
import os

def parse_gff3(filepath):
    """Parse GFF3 file and extract gene-level annotations."""
    genes = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            # Only extract gene-level features
            feature_type = parts[2]
            if feature_type != 'gene':
                continue
            
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]
            
            # Parse attributes to get gene name and biotype
            attr_dict = {}
            for attr in attributes.split(';'):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    attr_dict[key] = value
            
            gene_name = attr_dict.get('Name', attr_dict.get('gene', 'unknown'))
            gene_biotype = attr_dict.get('gene_biotype', 'unknown')
            
            genes.append({
                'gene_name': gene_name,
                'start': start,
                'end': end,
                'strand': strand,
                'biotype': gene_biotype,
                'length': end - start + 1
            })
    
    return pd.DataFrame(genes)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    gff3_path = os.path.join(project_dir, 'data', 'sequence.gff3')
    output_dir = os.path.join(project_dir, 'data', 'intermediate')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'genes.csv')
    
    print(f"Parsing GFF3: {gff3_path}")
    genes_df = parse_gff3(gff3_path)
    
    # Save
    genes_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(genes_df)} genes to {output_path}")
    
    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"Total genes: {len(genes_df)}")
    print(f"\nBiotype distribution:")
    print(genes_df['biotype'].value_counts().to_string())
    print(f"\nGenome coverage: {genes_df['start'].min()} - {genes_df['end'].max()}")
    print(f"\nFirst 5 genes:")
    print(genes_df.head().to_string())
    print(f"\nLast 5 genes:")
    print(genes_df.tail().to_string())
    
    # Check for D-loop region (not a gene, but important)
    print(f"\nNote: D-loop region (16024-16569) is not a gene and is not included.")

if __name__ == '__main__':
    main()
