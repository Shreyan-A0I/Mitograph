#!/usr/bin/env python3
"""
Phase A - Step 2: Parse ClinVar MT variant summary.
Reads clinvar_mt_variant_summary.csv and outputs a cleaned, deduplicated CSV
of single nucleotide variants with simplified clinical significance labels.

Output: data/intermediate/clinvar_variants.csv
"""

import pandas as pd
import os

def simplify_clinical_significance(sig):
    """Map ClinVar clinical significance to simplified labels."""
    if pd.isna(sig):
        return 'Other'
    sig_lower = sig.lower().strip()
    
    if 'pathogenic' in sig_lower and 'likely' not in sig_lower and 'conflict' not in sig_lower:
        return 'Pathogenic'
    elif 'likely pathogenic' in sig_lower:
        return 'Likely pathogenic'
    elif 'benign' in sig_lower and 'likely' not in sig_lower and 'conflict' not in sig_lower:
        return 'Benign'
    elif 'likely benign' in sig_lower:
        return 'Likely benign'
    elif 'uncertain significance' in sig_lower:
        return 'VUS'
    elif 'conflicting' in sig_lower:
        return 'Conflicting'
    else:
        return 'Other'

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    
    input_path = os.path.join(project_dir, 'data', 'clinvar_mt_variant_summary.csv')
    output_dir = os.path.join(project_dir, 'data', 'intermediate')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'clinvar_variants.csv')
    
    print(f"Reading ClinVar: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Raw rows: {len(df)}")
    
    # Filter to single nucleotide variants
    snv_df = df[df['Type'] == 'single nucleotide variant'].copy()
    print(f"After SNV filter: {len(snv_df)}")
    
    # Simplify clinical significance
    snv_df['clinical_significance'] = snv_df['ClinicalSignificance'].apply(simplify_clinical_significance)
    
    # Extract key columns and rename
    result = snv_df[[
        'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF',
        'clinical_significance', 'GeneSymbol', 'PhenotypeList',
        'Assembly', 'ReviewStatus', 'NumberSubmitters', '#AlleleID', 'VariationID'
    ]].copy()
    
    result.columns = [
        'pos', 'ref', 'alt', 'clinical_significance', 'gene_symbol',
        'phenotype_list', 'assembly', 'review_status', 'n_submitters',
        'allele_id', 'variation_id'
    ]
    
    # Deduplicate: keep GRCh38 when both assemblies present, otherwise keep first
    # First prefer GRCh38
    result_38 = result[result['assembly'] == 'GRCh38'].copy()
    result_37_only = result[
        (result['assembly'] == 'GRCh37') & 
        (~result['variation_id'].isin(result_38['variation_id']))
    ].copy()
    result_dedup = pd.concat([result_38, result_37_only], ignore_index=True)
    
    # Further deduplicate by (pos, ref, alt) - keep the one with more submitters
    result_dedup = result_dedup.sort_values('n_submitters', ascending=False)
    result_dedup = result_dedup.drop_duplicates(subset=['pos', 'ref', 'alt'], keep='first')
    result_dedup = result_dedup.sort_values('pos').reset_index(drop=True)
    
    # Drop assembly column (no longer needed)
    result_dedup = result_dedup.drop(columns=['assembly'])
    
    # Save
    result_dedup.to_csv(output_path, index=False)
    print(f"\nSaved {len(result_dedup)} deduplicated variants to {output_path}")
    
    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"Total unique variants: {len(result_dedup)}")
    print(f"\nClinical significance distribution:")
    print(result_dedup['clinical_significance'].value_counts().to_string())
    print(f"\nPosition range: {result_dedup['pos'].min()} - {result_dedup['pos'].max()}")
    print(f"\nTop 10 genes by variant count:")
    gene_counts = result_dedup['gene_symbol'].value_counts().head(10)
    print(gene_counts.to_string())
    print(f"\nSample rows:")
    print(result_dedup.head(10).to_string())

if __name__ == '__main__':
    main()
