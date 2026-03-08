#!/usr/bin/env python3
"""
Phase A - Step 3: Extract tables from MITOMAP PostgreSQL dump.
Parses COPY ... FROM stdin blocks directly from the SQL dump file
and writes each table to a clean CSV.

Output: data/intermediate/mitomap_*.csv
"""

import os
import csv
import io

# Tables to extract with their column definitions
TABLES_TO_EXTRACT = {
    'mitomap.mmutation': [
        'id', 'locus', 'dz', 'allele', 'position', 'refna', 'regna',
        'aa', 'cons', 'contr', 'homo', 'hetero', 'status', 'cfrm_date'
    ],
    'mitomap.rtmutation': [
        'id', 'locus', 'dz', 'allele', 'position', 'refna', 'regna',
        'rna', 'cons', 'contr', 'homo', 'hetero', 'status', 'cfrm_date'
    ],
    'mitomap.phenotype': [
        'id', 'short_name', 'name', 'url', 'note'
    ],
    'mitomap.locus': [
        'id', 'name', 'common_name', 'starting', 'ending',
        'strand', 'type', 'product', 'protein_id'
    ],
    'mitomap.mitotip': [
        'pos', 'ref', 'alt', 'score'
    ],
    'mitomap.apogee': [
        'id', 'position', 'refna', 'regna', 'score', 'status'
    ],
    'mitomap.hmtvar': [
        'id', 'pos', 'ref', 'alt', 'locus', 'disease_score', 'pathogenicity',
        'model', 'model_position', 'strutt_3', 'stem_loop', 'dbsnp',
        'mamit_trna', 'mitomap_homo', 'mitomap_hetero'
    ],
}

def extract_tables_from_dump(dump_path, output_dir):
    """
    Parse PostgreSQL dump to extract COPY blocks for target tables.
    This reads the file line-by-line to handle the 460MB dump efficiently.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Track which table we're currently reading
    current_table = None
    current_writer = None
    current_file = None
    current_columns = None
    row_counts = {}
    
    print(f"Scanning dump file: {dump_path}")
    print(f"Looking for {len(TABLES_TO_EXTRACT)} tables...")
    
    with open(dump_path, 'r', encoding='utf-8', errors='replace') as f:
        line_num = 0
        for line in f:
            line_num += 1
            
            if line_num % 5_000_000 == 0:
                print(f"  ... processed {line_num:,} lines")
            
            # Check for COPY statement
            if line.startswith('COPY '):
                # Parse: COPY schema.table (col1, col2, ...) FROM stdin;
                # Extract table name
                parts = line.split()
                if len(parts) >= 2:
                    table_name = parts[1]
                    
                    if table_name in TABLES_TO_EXTRACT:
                        current_table = table_name
                        current_columns = TABLES_TO_EXTRACT[table_name]
                        
                        # Extract short name for filename
                        short_name = table_name.replace('mitomap.', '')
                        out_path = os.path.join(output_dir, f'mitomap_{short_name}.csv')
                        current_file = open(out_path, 'w', newline='', encoding='utf-8')
                        current_writer = csv.writer(current_file)
                        current_writer.writerow(current_columns)
                        row_counts[table_name] = 0
                        print(f"  Found table: {table_name}")
                continue
            
            # If we're in a COPY block, read data lines
            if current_table is not None:
                # End of COPY block
                if line.strip() == '\\.':
                    print(f"    -> {row_counts[current_table]:,} rows extracted")
                    current_file.close()
                    current_table = None
                    current_writer = None
                    current_file = None
                    continue
                
                # Parse tab-separated data
                # PostgreSQL COPY uses tab-separated values with \N for NULL
                values = line.rstrip('\n').split('\t')
                # Replace \N with empty string
                values = ['' if v == '\\N' else v for v in values]
                
                # Only take as many values as we have columns
                values = values[:len(current_columns)]
                
                current_writer.writerow(values)
                row_counts[current_table] += 1
    
    # Close any remaining open file
    if current_file is not None:
        current_file.close()
    
    return row_counts

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    
    dump_path = os.path.join(project_dir, 'data', 'mitomap.dump.sql')
    output_dir = os.path.join(project_dir, 'data', 'intermediate')
    
    row_counts = extract_tables_from_dump(dump_path, output_dir)
    
    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    print(f"Tables extracted: {len(row_counts)}")
    for table, count in row_counts.items():
        print(f"  {table}: {count:,} rows")
    
    # Quick peek at each extracted file
    import pandas as pd
    for table_name in row_counts:
        short_name = table_name.replace('mitomap.', '')
        csv_path = os.path.join(output_dir, f'mitomap_{short_name}.csv')
        df = pd.read_csv(csv_path, nrows=5)
        print(f"\n--- {short_name} (first 5 rows) ---")
        print(df.to_string())

if __name__ == '__main__':
    main()
