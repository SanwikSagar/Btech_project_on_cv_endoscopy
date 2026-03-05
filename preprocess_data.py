import os
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

def preprocess_images(data_dir, output_dir, target_size=224):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata_file = data_path / "metadata.csv"
    if not metadata_file.exists():
        print(f"Error: metadata.csv not found in {data_dir}")
        return
    
    df = pd.read_csv(metadata_file)
    
    print(f"Preprocessing {len(df)} images...")
    print(f"Target size: {target_size}x{target_size}")
    
    processed_metadata = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_id = row["image_id"]
        class_name = row["class"]
        
        input_path = data_path / "images" / class_name / image_id
        output_class_dir = output_path / "images" / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_class_dir / image_id
        
        processed_metadata.append({
            "image_id": image_id,
            "class": class_name,
            "path": str(output_file.relative_to(output_path)),
            "split": row.get("split", "unassigned")
        })
    
    processed_df = pd.DataFrame(processed_metadata)
    processed_df.to_csv(output_path / "metadata.csv", index=False)
    
    print(f"\nPreprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total images processed: {len(processed_df)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_size", type=int, default=224)
    args = parser.parse_args()
    
    preprocess_images(args.data_dir, args.output_dir, args.target_size)

if __name__ == "__main__":
    main()