import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

def create_stratified_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    data_path = Path(data_dir)
    metadata_file = data_path / "metadata.csv"
    
    if not metadata_file.exists():
        print(f"Error: metadata.csv not found in {data_dir}")
        return
    
    df = pd.read_csv(metadata_file)
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        print("Error: Split ratios must sum to 1.0")
        return
    
    print(f"Creating splits: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print(f"Total samples: {len(df)}")
    
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_ratio), 
        stratify=df['class'],
        random_state=42
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df['class'],
        random_state=42
    )
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    output_dir = data_path / "splits"
    output_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    print(f"\nClass distribution:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        print(split_df['class'].value_counts().sort_index())
    
    print(f"\nSplits saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()
    
    create_stratified_splits(
        args.data_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == "__main__":
    main()