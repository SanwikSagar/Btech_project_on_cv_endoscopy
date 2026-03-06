import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import argparse
from tqdm import tqdm

CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]


def create_model(num_classes, architecture='resnet18'):
    arch = architecture.lower()
    if arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path, num_classes):
    checkpoint = torch.load(model_path, map_location='cpu')
    architecture = checkpoint.get('architecture', 'resnet18')
    print(f"Loading {architecture} from {model_path}")

    model = create_model(num_classes, architecture)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms(image_size=224):
    """Test-Time Augmentation — average over 5 augmented versions for better predictions."""
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return [
        transforms.Compose(base),  # original
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomHorizontalFlip(p=1.0)] + base[1:]),
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomVerticalFlip(p=1.0)] + base[1:]),
        transforms.Compose([transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                             transforms.CenterCrop(image_size)] + base[1:]),
        transforms.Compose([transforms.Resize((image_size, image_size)),
                             transforms.RandomRotation(degrees=(90, 90))] + base[1:]),
    ]


def predict_single(model, image, transforms_list, device, use_tta=True):
    """Run inference, optionally with TTA."""
    if use_tta:
        all_probs = []
        for t in transforms_list:
            img_tensor = t(image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                all_probs.append(probs.cpu().numpy())
        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = np.argmax(avg_probs)
        confidence = float(avg_probs[0][pred_idx])
    else:
        transform = transforms_list[0]
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_tensor)
            probs = torch.nn.functional.softmax(out, dim=1)
            confidence, pred_idx_t = torch.max(probs, 1)
        pred_idx = pred_idx_t.item()
        confidence = float(confidence.item())

    return CLASSES[pred_idx], confidence


def generate_predictions(model_path, test_csv, data_root, output_file, use_tta=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(model_path, len(CLASSES))
    model = model.to(device)

    tta_transforms = get_tta_transforms()
    test_df = pd.read_csv(test_csv)
    data_root = Path(data_root)

    predictions = []
    missing_files = []

    print(f"Generating predictions for {len(test_df)} images (TTA={'on' if use_tta else 'off'})...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_id = row['image_id']
        image_path = data_root / row['path']

        if not image_path.exists():
            missing_files.append(str(image_path))
            # Fallback
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"\nWarning: Could not load {image_path}: {e}")
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        predicted_class, confidence = predict_single(model, image, tta_transforms, device, use_tta)

        predictions.append({
            "image_id": image_id,
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    if missing_files:
        print(f"\n  {len(missing_files)} image files were not found. Check your data_root path.")

    result = {
        "model_name": f"HyperKvasir Classifier (TTA={'on' if use_tta else 'off'})",
        "predictions": predictions
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n Predictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    args = parser.parse_args()

    generate_predictions(
        args.model_path,
        args.test_csv,
        args.data_root,
        args.output,
        use_tta=not args.no_tta
    )


if __name__ == "__main__":
    main()