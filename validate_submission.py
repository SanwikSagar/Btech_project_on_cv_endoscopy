import json
import argparse

ALLOWED_CLASSES = {
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
}

def validate_submission(submission_file):
    print("Validating submission file...")
    
    try:
        with open(submission_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except FileNotFoundError:
        print(f"Error: File {submission_file} not found")
        return False
    
    if "predictions" not in data:
        print("Error: Missing 'predictions' key in JSON")
        return False
    
    if not isinstance(data["predictions"], list):
        print("Error: 'predictions' must be a list")
        return False
    
    if len(data["predictions"]) == 0:
        print("Error: No predictions found")
        return False
    
    image_ids = set()
    
    for i, pred in enumerate(data["predictions"]):
        if "image_id" not in pred:
            print(f"Error: Prediction {i} missing 'image_id'")
            return False
        
        if "predicted_class" not in pred:
            print(f"Error: Prediction {i} missing 'predicted_class'")
            return False
        
        if "confidence" not in pred:
            print(f"Error: Prediction {i} missing 'confidence'")
            return False
        
        if pred["image_id"] in image_ids:
            print(f"Error: Duplicate image_id found: {pred['image_id']}")
            return False
        
        image_ids.add(pred["image_id"])
        
        if pred["predicted_class"] not in ALLOWED_CLASSES:
            print(f"Error: Invalid class '{pred['predicted_class']}' in prediction {i}")
            print(f"Allowed classes: {ALLOWED_CLASSES}")
            return False
        
        confidence = pred["confidence"]
        if not isinstance(confidence, (int, float)):
            print(f"Error: Confidence must be numeric in prediction {i}")
            return False
        
        if not (0.0 <= confidence <= 1.0):
            print(f"Error: Confidence must be between 0 and 1 in prediction {i}")
            return False
    
    print("\n" + "="*60)
    print("VALIDATION PASSED")
    print("="*60)
    print(f"Total predictions: {len(data['predictions'])}")
    print(f"Unique images: {len(image_ids)}")
    print("\nYour submission is ready for evaluation!")
    print("="*60 + "\n")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, required=True)
    args = parser.parse_args()
    
    validate_submission(args.submission)

if __name__ == "__main__":
    main()