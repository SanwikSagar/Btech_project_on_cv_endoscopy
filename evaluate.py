import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
import argparse

CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]

def load_ground_truth(test_csv):
    df = pd.read_csv(test_csv)
    ground_truth = {}
    for _, row in df.iterrows():
        ground_truth[row['image_id']] = row['class']
    return ground_truth

def load_predictions(pred_file):
    with open(pred_file, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    for pred in data['predictions']:
        predictions[pred['image_id']] = pred['predicted_class']
    
    return predictions

def evaluate_predictions(ground_truth, predictions):
    y_true = []
    y_pred = []
    
    for image_id in ground_truth:
        if image_id not in predictions:
            print(f"Warning: Missing prediction for {image_id}")
            continue
        
        y_true.append(ground_truth[image_id])
        y_pred.append(predictions[image_id])
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    
    competition_score = (macro_f1 * 0.5) + (bal_acc * 0.5)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nMacro F1-Score:        {macro_f1:.4f}")
    print(f"Balanced Accuracy:     {bal_acc:.4f}")
    print(f"Macro Precision:       {macro_precision:.4f}")
    print(f"Macro Recall:          {macro_recall:.4f}")
    
    print(f"\n{'='*60}")
    print(f"COMPETITION SCORE:     {competition_score:.4f}")
    print(f"{'='*60}\n")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES)
    print(report)
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    print(f"\n{'':15s}", end="")
    for cls in CLASSES:
        print(f"{cls[:8]:>8s}", end=" ")
    print()
    
    for i, cls in enumerate(CLASSES):
        print(f"{cls:15s}", end=" ")
        for j in range(len(CLASSES)):
            print(f"{cm[i,j]:7d}", end=" ")
        print()
    
    return {
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'competition_score': competition_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()
    
    ground_truth = load_ground_truth(args.test_csv)
    predictions = load_predictions(args.predictions)
    
    results = evaluate_predictions(ground_truth, predictions)

if __name__ == "__main__":
    main()