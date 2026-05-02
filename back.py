"""
============================================================
Emotion Detection CNN - Training Script
Dataset: Face Expression Recognition (Kaggle - jonathanoheix)
GPU: CPU Fallback (Waiting for RTX 5060 PyTorch support)
============================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    "data_dir":      "data",          
    "model_save":    "best_model.pth",
    "img_size":      96,              
    "batch_size":    64,
    "num_epochs":    50,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "label_smooth":  0.1,
    "mixup_alpha":   0.4,
    "num_workers":   0,               # Changed to 0 to prevent Windows crashes
    "seed":          42,
    "emotions":      ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
}

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
# Force CPU for now (RTX 5060 sm_120 not yet in stable PyTorch)
device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

# ═══════════════════════════════════════════
# 1. DATA PREPROCESSING & AUGMENTATION
# ═══════════════════════════════════════════

IMG_SIZE = CONFIG["img_size"]

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_datasets(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "validation")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    # Compute class weights for imbalanced FER dataset
    class_counts = np.array([len(os.listdir(os.path.join(train_dir, c)))
                              for c in train_ds.classes])
    total = class_counts.sum()
    class_weights = torch.tensor(total / (len(train_ds.classes) * class_counts),
                                 dtype=torch.float32).to(device)

    print(f"\n[DATA] Classes : {train_ds.classes}")
    print(f"[DATA] Train   : {len(train_ds):,} images")
    print(f"[DATA] Val     : {len(val_ds):,} images")
    print(f"[DATA] Class counts (train): {class_counts}")

    # persistent_workers=False automatically when num_workers=0
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"],
                              pin_memory=False) 
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=CONFIG["num_workers"],
                              pin_memory=False)
    return train_loader, val_loader, train_ds.classes, class_weights


# ═══════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ═══════════════════════════════════════════

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if "features.0" in name or "features.1" in name or \
               "features.2" in name or "features.3" in name:
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ═══════════════════════════════════════════
# 3. MIXUP AUGMENTATION
# ═══════════════════════════════════════════

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════
# 4. TRAINING LOOP
# ═══════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        imgs, y_a, y_b, lam = mixup_data(imgs, labels, CONFIG["mixup_alpha"])
        optimizer.zero_grad(set_to_none=True)

        # Only use autocast if on GPU
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(y_a).float() +
                    (1 - lam) * predicted.eq(y_b).float()).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# ═══════════════════════════════════════════
# 5. MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════

def train():
    train_loader, val_loader, classes, class_weights = load_datasets(CONFIG["data_dir"])
    num_classes = len(classes)

    model = EmotionNet(num_classes=num_classes).to(device)
    print(f"\n[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[MODEL] Trainable : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["label_smooth"]
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    # Setup AMP safely (only for CUDA)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "═" * 60)
    print("  TRAINING START")
    print("═" * 60)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, use_amp)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, use_amp)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        flag = " ◀ BEST" if val_acc > best_val_acc else ""
        print(f"Epoch [{epoch:3d}/{CONFIG['num_epochs']}] "
              f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
              f"| {elapsed:.1f}s{flag}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "classes": classes,
                "img_size": IMG_SIZE,
            }, CONFIG["model_save"])

    print(f"\n[DONE] Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"[DONE] Model saved to:    {CONFIG['model_save']}")

    # ── Final evaluation on best model ──
    checkpoint = torch.load(CONFIG["model_save"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion, use_amp)

    print("\n" + "═" * 60)
    print("  FINAL CLASSIFICATION REPORT (Validation Set)")
    print("═" * 60)
    print(classification_report(final_labels, final_preds, target_names=classes, digits=4))

    metrics = {
        "accuracy":  accuracy_score(final_labels, final_preds),
        "precision": precision_score(final_labels, final_preds, average="weighted"),
        "recall":    recall_score(final_labels, final_preds, average="weighted"),
        "f1":        f1_score(final_labels, final_preds, average="weighted"),
    }
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")

    plot_training_history(history)
    plot_confusion_matrix(final_labels, final_preds, classes)

    return model, history, metrics


# ═══════════════════════════════════════════
# 6. VISUALIZATIONS
# ═══════════════════════════════════════════

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=3, label="Val Loss")
    ax1.set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-o", markersize=3, label="Val Acc")
    ax2.set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved: training_history.png")


def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_ylabel("True Label"); axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved: confusion_matrix.png")


# ═══════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════

if __name__ == "__main__":
    train()