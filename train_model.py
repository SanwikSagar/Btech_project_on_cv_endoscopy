import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml
import argparse
from sklearn.metrics import f1_score, balanced_accuracy_score

CLASSES = [
    "normal", "polyp", "ulcer", "bleeding", "inflammation",
    "erosion", "tumor", "stricture", "diverticula", "foreign_body"
]

class EndoscopyDataset(Dataset):
    def __init__(self, csv_file, data_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_root / row['path']
        label = self.class_to_idx[row['class']]
        
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(image_size, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def create_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_f1, epoch_bal_acc

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_transform = get_transforms(config['image_size'], is_training=True)
    val_transform = get_transforms(config['image_size'], is_training=False)
    
    data_root = Path(config['data_dir'])
    train_dataset = EndoscopyDataset(
        data_root / "splits/train.csv",
        data_root,
        train_transform
    )
    val_dataset = EndoscopyDataset(
        data_root / "splits/val.csv",
        data_root,
        val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    model = create_model(len(CLASSES), pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    best_f1 = 0.0
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_bal_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Bal Acc: {val_bal_acc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_bal_acc': val_bal_acc,
            }, models_dir / "best_model.pth")
            print(f"Saved best model with F1: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best validation F1: {best_f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config)

if __name__ == "__main__":
    main()