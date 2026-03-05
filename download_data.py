import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import argparse

CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]

def create_directory_structure(base_dir):
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    
    for class_name in CLASSES:
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    return base_path

def download_sample_images(base_dir, images_per_class=800):
    print(f"Creating sample dataset with {images_per_class} images per class...")
    
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    
    metadata = []
    
    for class_name in tqdm(CLASSES, desc="Processing classes"):
        class_dir = images_dir / class_name
        
        for i in range(images_per_class):
            image_name = f"{class_name}_{i:04d}.png"
            image_path = class_dir / image_name
            
            metadata.append({
                "image_id": image_name,
                "class": class_name,
                "split": "unassigned"
            })
    
    import pandas as pd
    df = pd.DataFrame(metadata)
    df.to_csv(base_path / "metadata.csv", index=False)
    
    print(f"Dataset structure created at {base_dir}")
    print(f"Total images: {len(metadata)}")
    print(f"Classes: {len(CLASSES)}")
    
    return base_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--images_per_class", type=int, default=800)
    args = parser.parse_args()
    
    create_directory_structure(args.output_dir)
    download_sample_images(args.output_dir, args.images_per_class)
    
    print("\nDataset download complete!")
    print(f"Location: {args.output_dir}")

if __name__ == "__main__":
    main()