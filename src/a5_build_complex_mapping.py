#!/usr/bin/env python3
"""
Phase A - Step 5: Build Gene-to-Complex mapping.
Maps mitochondrial protein-coding genes to their respiratory chain complexes.

Complex I:   ND1, ND2, ND3, ND4, ND4L, ND5, ND6 (NADH dehydrogenase)
Complex III: CYTB (Cytochrome b)
Complex IV:  COX1, COX2, COX3 (Cytochrome c oxidase)
Complex V:   ATP6, ATP8 (ATP synthase)

Output: data/intermediate/gene_complex_mapping.csv
"""

import pandas as pd
import os

# Respiratory chain complex mapping (standard mitochondrial biology)
GENE_COMPLEX_MAP = {
    # Complex I - NADH dehydrogenase
    'ND1': 'I', 'ND2': 'I', 'ND3': 'I', 'ND4': 'I', 
    'ND4L': 'I', 'ND5': 'I', 'ND6': 'I',
    # Complex III - Cytochrome bc1
    'CYTB': 'III',
    # Complex IV - Cytochrome c oxidase
    'COX1': 'IV', 'COX2': 'IV', 'COX3': 'IV',
    # Complex V - ATP synthase
    'ATP6': 'V', 'ATP8': 'V',
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    # Load genes from Step A1
    genes_path = os.path.join(intermediate_dir, 'genes.csv')
    genes = pd.read_csv(genes_path)
    print(f"Loaded {len(genes)} genes")
    
    # Map genes to complexes
    genes['complex'] = genes['gene_name'].map(GENE_COMPLEX_MAP)
    genes['complex'] = genes['complex'].fillna('none')
    
    # Create the mapping table
    mapping = genes[['gene_name', 'biotype', 'complex']].copy()
    
    # Save
    output_path = os.path.join(intermediate_dir, 'gene_complex_mapping.csv')
    mapping.to_csv(output_path, index=False)
    print(f"Saved {len(mapping)} gene-complex mappings to {output_path}")
    
    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"\nComplex distribution:")
    print(mapping['complex'].value_counts().to_string())
    print(f"\nProtein-coding genes in complexes:")
    complex_genes = mapping[mapping['complex'] != 'none']
    for _, row in complex_genes.iterrows():
        print(f"  {row['gene_name']} -> Complex {row['complex']}")
    
    print(f"\nGenes without complex assignment (tRNA/rRNA):")
    no_complex = mapping[mapping['complex'] == 'none']
    for _, row in no_complex.iterrows():
        print(f"  {row['gene_name']} ({row['biotype']})")

if __name__ == '__main__':
    main()
