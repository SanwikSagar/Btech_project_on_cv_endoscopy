import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



class EndoscopyDataset(Dataset):
    def __init__(self, csv_file, data_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        # Filter out classes not in our list
        valid_mask = self.df['class'].isin(self.class_to_idx)
        removed = (~valid_mask).sum()
        if removed > 0:
            print(f"  [Dataset] Removed {removed} rows with unknown classes")
        self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.data_root / row['path']
        label = self.class_to_idx[row['class']]

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"  Warning: Could not load {image_path}: {e}. Using blank image.")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency weights for WeightedRandomSampler."""
        counts = self.df['class'].value_counts()
        weights = []
        for _, row in self.df.iterrows():
            weights.append(1.0 / counts[row['class']])
        return weights

def get_transforms(image_size, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
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

def create_model(num_classes, architecture='resnet18', pretrained=True):
    arch = architecture.lower()
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose: resnet18, resnet50, efficientnet_b0, efficientnet_b3")
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
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
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return epoch_loss, epoch_f1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    epoch_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_f1, epoch_bal_acc

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("  No GPU detected. Training will be slow. Consider using Google Colab for GPU.")

    data_root = Path(config['data_dir'])
    image_size = config.get('image_size', 224)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 50)
    lr = config.get('learning_rate', 0.001)
    architecture = config.get('model_architecture', 'resnet18')
    pretrained = config.get('pretrained', True)
    num_workers = config.get('num_workers', 0)  # 0 is safest on Windows

    # Datasets
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)

    train_dataset = EndoscopyDataset(data_root / "splits/train.csv", data_root, train_transform)
    val_dataset = EndoscopyDataset(data_root / "splits/val.csv", data_root, val_transform)

    print(f"\nDataset sizes → Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Weighted sampler to handle class imbalance
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # Model
    print(f"\nBuilding model: {architecture} (pretrained={pretrained})")
    model = create_model(len(CLASSES), architecture, pretrained)
    model = model.to(device)

    # Focal loss (better for imbalanced medical datasets)
    criterion = FocalLoss(gamma=2.0)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    best_score = 0.0
    best_epoch = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 65)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]  LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_bal_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Competition score
        competition_score = (val_f1 * 0.5) + (val_bal_acc * 0.5)

        print(f"  Train → Loss: {train_loss:.4f}  F1: {train_f1:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}  F1: {val_f1:.4f}  BalAcc: {val_bal_acc:.4f}  Score: {competition_score:.4f}")

        if competition_score > best_score:
            best_score = competition_score
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_bal_acc': val_bal_acc,
                'competition_score': competition_score,
                'classes': CLASSES,
                'architecture': architecture,
            }, models_dir / "best_model.pth")
            print(f"  New best model saved! Score: {best_score:.4f}")

    print(f"\n{'='*65}")
    print(f"Training complete! Best score: {best_score:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {models_dir / 'best_model.pth'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_model(config)


if __name__ == "__main__":
    main()