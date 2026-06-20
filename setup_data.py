"""
FraudShield-Lite — Data Setup Script

Downloads PaySim dataset from Kaggle, extracts 500K row subset,
and generates the feature-engineered dataset.

Usage:
    python setup_data.py              # Full setup (download + features)
    python setup_data.py --skip-download  # Skip download, use existing CSV
"""
import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_CSV = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
SUBSET_CSV = os.path.join(DATA_DIR, "subset_500k.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1"


def download_paysim():
    """Download PaySim dataset from Kaggle."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(RAW_CSV):
        print(f"  Raw CSV already exists: {RAW_CSV}")
        return
    
    zip_path = os.path.join(DATA_DIR, "paysim.zip")
    print(f"  Downloading from Kaggle...")
    
    result = subprocess.run(
        ["curl", "-L", "-o", zip_path, KAGGLE_URL],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        print(f"  ❌ Download failed: {result.stderr}")
        print(f"  Manual download: {KAGGLE_URL}")
        sys.exit(1)
    
    # Unzip
    subprocess.run(["unzip", "-o", zip_path, "-d", DATA_DIR], check=True)
    os.remove(zip_path)
    print(f"  ✅ Downloaded and extracted")


def create_subset(n_rows=500_000):
    """Create stratified subset from full PaySim data."""
    if os.path.exists(SUBSET_CSV):
        print(f"  Subset already exists: {SUBSET_CSV}")
        return
    
    print(f"  Reading {RAW_CSV}...")
    chunksize = 200_000
    chunks_sampled = []
    total_kept = 0
    
    for i, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=chunksize)):
        n_sample = int(len(chunk) * 0.08)
        sampled = chunk.sample(n=min(n_sample, len(chunk)), random_state=42)
        chunks_sampled.append(sampled)
        total_kept += len(sampled)
        
        print(f"  Chunk {i}: kept {len(sampled):,} (total: {total_kept:,})")
        
        if total_kept >= n_rows:
            break
    
    df = pd.concat(chunks_sampled, ignore_index=True)
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    
    df = df.sort_values("step").reset_index(drop=True)
    df.to_csv(SUBSET_CSV, index=False)
    
    # Cleanup raw
    if os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)
    
    print(f"  ✅ Subset created: {df.shape}")
    print(f"     Fraud: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.3f}%)")


def create_features():
    """Generate features from subset."""
    if os.path.exists(FEATURES_CSV):
        print(f"  Features already exist: {FEATURES_CSV}")
        return
    
    sys.path.insert(0, os.path.dirname(__file__))
    from src.features import create_all_features
    
    print(f"  Loading {SUBSET_CSV}...")
    df = pd.read_csv(SUBSET_CSV)
    
    print(f"  Engineering features...")
    features = create_all_features(df)
    
    features.to_csv(FEATURES_CSV, index=False)
    print(f"  ✅ Features created: {features.shape}")


def main():
    parser = argparse.ArgumentParser(description="Setup FraudShield-Lite data")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FraudShield-Lite — Data Setup")
    print("=" * 60)
    
    # Step 1: Download
    print("\n[1/3] Download PaySim dataset")
    if args.skip-download:
        print("  Skipped (--skip-download)")
        if not os.path.exists(SUBSET_CSV):
            print(f"  ❌ {SUBSET_CSV} not found. Run without --skip-download first.")
            sys.exit(1)
    else:
        download_paysim()
    
    # Step 2: Create subset
    print("\n[2/3] Create 500K subset")
    create_subset()
    
    # Step 3: Feature engineering
    print("\n[3/3] Feature engineering")
    create_features()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print(f"  {SUBSET_CSV}")
    print(f"  {FEATURES_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
