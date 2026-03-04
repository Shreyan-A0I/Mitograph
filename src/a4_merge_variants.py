#!/usr/bin/env python3
"""
Phase A - Step 4: Merge variants from ClinVar and MITOMAP.
Joins ClinVar and MITOMAP variant data on (position, ref, alt).
Adds PhyloP conservation scores from conservation_scores.txt.
Adds MITOMAP scores (mitotip/apogee) as secondary features.

Output: data/intermediate/merged_variants.csv
"""

import pandas as pd
import os


def load_phylop_scores(filepath):
    """
    Parses a UCSC Wiggle (.txt) file to extract basewise conservation scores.
    Returns a dictionary mapping integer positions to float scores.
    """
    conservation_dict = {}
    print(f"Loading PhyloP scores from {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip all UCSC metadata headers
            if line.startswith(('track', '#', 'variableStep')):
                continue
            
            # Split the line by whitespace
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    pos = int(parts[0])
                    score = float(parts[1])
                    conservation_dict[pos] = score
                except ValueError:
                    continue
                
    print(f"Successfully loaded {len(conservation_dict)} conservation scores.")
    return conservation_dict


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    intermediate_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    # --- Load ClinVar variants ---
    clinvar_path = os.path.join(intermediate_dir, 'clinvar_variants.csv')
    clinvar = pd.read_csv(clinvar_path)
    print(f"ClinVar variants: {len(clinvar)}")
    
    # --- Load MITOMAP mutations ---
    mmut_path = os.path.join(intermediate_dir, 'mitomap_mmutation.csv')
    rtmut_path = os.path.join(intermediate_dir, 'mitomap_rtmutation.csv')
    
    mmut = pd.read_csv(mmut_path)
    rtmut = pd.read_csv(rtmut_path)
    print(f"MITOMAP coding mutations (mmutation): {len(mmut)}")
    print(f"MITOMAP tRNA/rRNA mutations (rtmutation): {len(rtmut)}")
    
    # Standardize MITOMAP mutations
    # mmutation: position, refna, regna, dz (disease), locus, status
    mmut_std = mmut[['position', 'refna', 'regna', 'dz', 'locus', 'status']].copy()
    mmut_std.columns = ['pos', 'ref', 'alt', 'mitomap_disease', 'mitomap_locus', 'mitomap_status']
    mmut_std['mutation_type'] = 'coding'
    
    # rtmutation: position, refna, regna, dz, locus, status
    rtmut_std = rtmut[['position', 'refna', 'regna', 'dz', 'locus', 'status']].copy()
    rtmut_std.columns = ['pos', 'ref', 'alt', 'mitomap_disease', 'mitomap_locus', 'mitomap_status']
    rtmut_std['mutation_type'] = 'tRNA/rRNA'
    
    # Combine MITOMAP
    mitomap_all = pd.concat([mmut_std, rtmut_std], ignore_index=True)
    mitomap_all['pos'] = pd.to_numeric(mitomap_all['pos'], errors='coerce')
    mitomap_all = mitomap_all.dropna(subset=['pos'])
    mitomap_all['pos'] = mitomap_all['pos'].astype(int)
    print(f"Combined MITOMAP mutations: {len(mitomap_all)}")
    
    # Deduplicate MITOMAP by (pos, ref, alt) - aggregate diseases
    mitomap_grouped = mitomap_all.groupby(['pos', 'ref', 'alt']).agg({
        'mitomap_disease': lambda x: '|'.join(sorted(set(str(d) for d in x if pd.notna(d) and str(d).strip()))),
        'mitomap_locus': 'first',
        'mitomap_status': 'first',
        'mutation_type': 'first'
    }).reset_index()
    print(f"MITOMAP after dedup: {len(mitomap_grouped)}")
    
    # --- Merge ClinVar + MITOMAP ---
    merged = clinvar.merge(mitomap_grouped, on=['pos', 'ref', 'alt'], how='outer', indicator=True)
    print(f"\nMerge results:")
    print(merged['_merge'].value_counts().to_string())
    
    # Fill clinical_significance for MITOMAP-only variants
    merged.loc[merged['_merge'] == 'right_only', 'clinical_significance'] = 'MITOMAP_only'
    merged = merged.drop(columns=['_merge'])
    
    # --- Load PhyloP conservation scores ---
    phylop_path = os.path.join(project_dir, 'data', 'conservation_scores.txt')
    phylop_dict = load_phylop_scores(phylop_path)
    
    # Add PhyloP score for each variant position
    merged['phylop_score'] = merged['pos'].map(phylop_dict)
    phylop_coverage = merged['phylop_score'].notna().sum()
    print(f"\nPhyloP score coverage: {phylop_coverage}/{len(merged)} ({100*phylop_coverage/len(merged):.1f}%)")
    
    # --- Load MITOMAP scores (mitotip, apogee) ---
    mitotip_path = os.path.join(intermediate_dir, 'mitomap_mitotip.csv')
    apogee_path = os.path.join(intermediate_dir, 'mitomap_apogee.csv')
    
    if os.path.exists(mitotip_path):
        mitotip = pd.read_csv(mitotip_path)
        mitotip = mitotip.rename(columns={'score': 'mitotip_score'})
        mitotip['pos'] = pd.to_numeric(mitotip['pos'], errors='coerce')
        mitotip = mitotip.dropna(subset=['pos'])
        mitotip['pos'] = mitotip['pos'].astype(int)
        # Deduplicate
        mitotip = mitotip.drop_duplicates(subset=['pos', 'ref', 'alt'], keep='first')
        merged = merged.merge(mitotip[['pos', 'ref', 'alt', 'mitotip_score']], 
                              on=['pos', 'ref', 'alt'], how='left')
        print(f"MitoTIP scores matched: {merged['mitotip_score'].notna().sum()}")
    
    if os.path.exists(apogee_path):
        apogee = pd.read_csv(apogee_path)
        apogee = apogee.rename(columns={'position': 'pos', 'refna': 'ref', 'regna': 'alt', 'score': 'apogee_score'})
        apogee['pos'] = pd.to_numeric(apogee['pos'], errors='coerce')
        apogee = apogee.dropna(subset=['pos'])
        apogee['pos'] = apogee['pos'].astype(int)
        apogee = apogee.drop_duplicates(subset=['pos', 'ref', 'alt'], keep='first')
        merged = merged.merge(apogee[['pos', 'ref', 'alt', 'apogee_score']], 
                              on=['pos', 'ref', 'alt'], how='left')
        print(f"APOGEE scores matched: {merged['apogee_score'].notna().sum()}")
    
    # Sort by position
    merged = merged.sort_values('pos').reset_index(drop=True)
    
    # Save
    output_path = os.path.join(intermediate_dir, 'merged_variants.csv')
    merged.to_csv(output_path, index=False)
    print(f"\nSaved {len(merged)} merged variants to {output_path}")
    
    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"Total merged variants: {len(merged)}")
    print(f"\nClinical significance distribution:")
    print(merged['clinical_significance'].value_counts().to_string())
    print(f"\nPhyloP score stats:")
    print(merged['phylop_score'].describe().to_string())
    print(f"\nPosition range: {merged['pos'].min()} - {merged['pos'].max()}")
    print(f"\nColumns: {list(merged.columns)}")

if __name__ == '__main__':
    main()
