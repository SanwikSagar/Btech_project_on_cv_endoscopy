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

def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location='cpu')
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

def generate_predictions(model_path, test_csv, data_root, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(model_path, len(CLASSES))
    model = model.to(device)
    
    transform = get_transform()
    
    test_df = pd.read_csv(test_csv)
    data_root = Path(data_root)
    
    predictions = []
    
    print(f"Generating predictions for {len(test_df)} images...")
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_id = row['image_id']
        image_path = data_root / row['path']
        
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASSES[pred_idx.item()]
        confidence_score = confidence.item()
        
        predictions.append({
            "image_id": image_id,
            "predicted_class": predicted_class,
            "confidence": float(confidence_score)
        })
    
    result = {
        "model_name": "ResNet18 Baseline",
        "predictions": predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nPredictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()
    
    generate_predictions(
        args.model_path,
        args.test_csv,
        args.data_root,
        args.output
    )

if __name__ == "__main__":
    main()