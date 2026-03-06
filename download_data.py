import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Mapping from HyperKvasir folder names to project's 10 class labels
# HyperKvasir has 23 classes; we map them to the closest relevant category.
HYPERKVASIR_CLASS_MAP = {
    # Pathological findings - direct matches
    "polyp": "polyp",
    "ulcerative-colitis-grade-0-1": "inflammation",
    "ulcerative-colitis-grade-1": "inflammation",
    "ulcerative-colitis-grade-1-2": "inflammation",
    "ulcerative-colitis-grade-2": "inflammation",
    "ulcerative-colitis-grade-2-3": "inflammation",
    "ulcerative-colitis-grade-3": "inflammation",
    "esophagitis-a": "ulcer",
    "esophagitis-b-d": "ulcer",
    "barretts": "ulcer",
    "barretts-short-segment": "ulcer",
    "hemorrhoids": "bleeding",
    "dyed-lifted-polyps": "polyp",
    "dyed-resection-margins": "polyp",

    # Anatomical landmarks - normal
    "cecum": "normal",
    "ileum": "normal",
    "retroflex-rectum": "normal",
    "hemorrhoids": "normal",
    "normal-pylorus": "normal",
    "normal-z-line": "normal",
    "normal-cecum": "normal",
    "retroflex-stomach": "normal",
    "z-line": "normal",
    "pylorus": "normal",

    # Quality / preparation
    "bbps-0-1": "normal",
    "bbps-2-3": "normal",
    "impacted-stool": "normal",

    # Therapeutic interventions
    "polyp-removal": "polyp",
}


TARGET_CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]


def find_all_images(kvasir_dir):
    """Walk the HyperKvasir directory tree and collect all images with their class."""
    kvasir_path = Path(kvasir_dir)
    records = []
    skipped = []

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for img_path in kvasir_path.rglob('*'):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # The parent folder name is the class
        class_folder = img_path.parent.name.lower()

        # Try exact match first, then partial match
        mapped_class = HYPERKVASIR_CLASS_MAP.get(class_folder)

        if mapped_class is None:
            # Try partial/fuzzy match
            for key, val in HYPERKVASIR_CLASS_MAP.items():
                if key in class_folder or class_folder in key:
                    mapped_class = val
                    break

        if mapped_class is None:
            skipped.append((str(img_path), class_folder))
            continue

        records.append({
            'original_path': str(img_path),
            'original_class': class_folder,
            'class': mapped_class,
            'filename': img_path.name
        })

    return records, skipped


def setup_dataset(kvasir_dir, output_dir, copy_images=True):
    print(f"Scanning HyperKvasir dataset at: {kvasir_dir}")
    records, skipped = find_all_images(kvasir_dir)

    print(f"Found {len(records)} images")
    if skipped:
        print(f"Skipped {len(skipped)} images with unrecognized classes:")
        for path, cls in skipped[:10]:
            print(f"  - {cls}: {path}")

    output_path = Path(output_dir)
    images_dir = output_path / "images"

    # Create class subdirectories
    for cls in TARGET_CLASSES:
        (images_dir / cls).mkdir(parents=True, exist_ok=True)

    metadata = []
    print(f"\n{'Copying' if copy_images else 'Linking'} images to {output_dir}...")

    for record in tqdm(records):
        src = Path(record['original_path'])
        cls = record['class']
        dest = images_dir / cls / src.name

        # Handle duplicate filenames
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while dest.exists():
                dest = images_dir / cls / f"{stem}_{counter}{suffix}"
                counter += 1

        if copy_images:
            shutil.copy2(src, dest)
        else:
            # Use symlinks to save disk space (Linux/Mac only)
            dest.symlink_to(src.resolve())

        rel_path = str(dest.relative_to(output_path))
        metadata.append({
            'image_id': dest.name,
            'class': cls,
            'path': rel_path,
            'split': 'unassigned'
        })

    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "metadata.csv", index=False)

    print(f"\n Dataset ready at: {output_dir}")
    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    print(df['class'].value_counts().sort_index().to_string())

    # Warn about missing classes
    present = set(df['class'].unique())
    missing = set(TARGET_CLASSES) - present
    if missing:
        print(f"\n  Warning: No images found for classes: {missing}")
        print("   These classes don't exist in HyperKvasir. Consider:")
        print("   - Removing them from CLASSES list, OR")
        print("   - Sourcing additional data for these categories")


def main():
    parser = argparse.ArgumentParser(description="Prepare HyperKvasir dataset for training")
    parser.add_argument("--kvasir_dir", type=str, required=True,
                        help="Path to extracted hyper-kvasir-labeled-images folder")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Where to save the prepared dataset")
    parser.add_argument("--no_copy", action="store_true",
                        help="Use symlinks instead of copying (saves disk space, Linux/Mac only)")
    args = parser.parse_args()

    setup_dataset(args.kvasir_dir, args.output_dir, copy_images=not args.no_copy)


if __name__ == "__main__":
    main()