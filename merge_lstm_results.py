#Created with Claude
"""
Merges chunk CSV files from LSTM grid search into one combined grid_search_results.csv.
Usage: python3 merge_lstm_results.py
"""
import pandas as pd
import glob
import os

RESULTS_DIR = "./results/LSTM"
CHUNK_PATTERN = os.path.join(RESULTS_DIR, "grid_search_results_chunk_*.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "grid_search_results.csv")

files = sorted(glob.glob(CHUNK_PATTERN))

if not files:
    print("No chunk files found. Nothing to merge.")
else:
    print(f"Found {len(files)} chunk files:")
    for f in files:
        print(f"  {f}")
    
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged into: {OUTPUT_FILE} ({len(df)} rows)")