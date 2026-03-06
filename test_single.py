import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import sys

# Must be 10 classes — matches how the model was trained
CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = models.resnet18(weights=None)  # fixed deprecation warning
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # 10 classes
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(image_url_or_path, model_path="./models/best_model.pth"):
    model = load_model(model_path)

    # Load from URL or local file
    if image_url_or_path.startswith("http"):
        response = requests.get(image_url_or_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_url_or_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    print("\n--- Prediction Results ---")
    for cls, prob in zip(CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls:15s} {prob:.2%}  {bar}")

    predicted = CLASSES[probs.argmax()]
    confidence = probs.max().item()
    print(f"\n  Predicted: {predicted.upper()} ({confidence:.2%} confidence)\n")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else input("Enter image URL or file path: ")
    predict(img)

# run>>
# py -3.11 test_single.py "local address or web address of image"