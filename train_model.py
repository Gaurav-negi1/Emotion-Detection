"""
============================================================
Emotion Detection CNN - MAXIMUM ACCURACY Version (FIXED)
Fixes:
  - OneCycleLR steps capped at swa_start epochs (no overflow)
  - Resume from checkpoint support
  - num_workers=0 to fix duplicate print spam on Windows
============================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.amp import GradScaler, autocast
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
    "swa_save":      "swa_model.pth",
    "resume":        "best_model.pth",   # set to None to train from scratch
    "img_size":      224,
    "batch_size":    32,
    "num_epochs":    80,
    "swa_start":     60,
    "swa_lr":        1e-5,
    "lr_backbone":   3e-5,
    "lr_head":       2e-4,
    "weight_decay":  1e-4,
    "label_smooth":  0.1,
    "mixup_alpha":   0.3,
    "cutmix_alpha":  1.0,
    "focal_gamma":   2.0,
    "tta_steps":     10,
    "num_workers":   0,   # 0 = no multiprocessing on Windows → fixes duplicate prints
    "seed":          42,
    "emotions":      ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    print(f"[INFO] Using device : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════
# 1. FOCAL LOSS
# ═══════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.weight          = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ═══════════════════════════════════════════
# 2. DATA TRANSFORMS
# ═══════════════════════════════════════════

IMG_SIZE = CONFIG["img_size"]

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                            scale=(0.85, 1.15), shear=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tta_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════
# 3. DATASET LOADER
# ═══════════════════════════════════════════

def load_datasets(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "validation")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    class_counts = np.array([
        len(os.listdir(os.path.join(train_dir, c))) for c in train_ds.classes
    ])
    total = class_counts.sum()
    class_weights = torch.tensor(
        total / (len(train_ds.classes) * class_counts), dtype=torch.float32
    ).to(device)

    print(f"\n[DATA] Classes      : {train_ds.classes}")
    print(f"[DATA] Train images : {len(train_ds):,}")
    print(f"[DATA] Val images   : {len(val_ds):,}")
    print(f"[DATA] Class counts : {class_counts}")
    print(f"[DATA] Class weights: {class_weights.cpu().numpy().round(2)}")

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=(CONFIG["num_workers"] == 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=(CONFIG["num_workers"] == 0)
    )
    return train_loader, val_loader, train_ds.classes, class_weights, val_dir


# ═══════════════════════════════════════════
# 4. MODEL
# ═══════════════════════════════════════════

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT
        )
        in_features = self.backbone.classifier[1].in_features  # 1792

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ═══════════════════════════════════════════
# 5. MIXUP + CUTMIX
# ═══════════════════════════════════════════

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════
# 6. TRAIN / EVAL
# ═══════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    scheduler, swa_active):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        r = np.random.rand()
        if r < 0.4:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, CONFIG["mixup_alpha"])
        elif r < 0.7:
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CONFIG["cutmix_alpha"])
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            outputs = model(imgs)
            loss    = mixed_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── KEY FIX: only step OneCycleLR before SWA starts ──
        if not swa_active:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(y_a).float() +
                    (1 - lam) * predicted.eq(y_b).float()).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct  += predicted.eq(labels).sum().item()
        total    += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


@torch.no_grad()
def evaluate_with_tta(model, val_dir, criterion, classes, n=10):
    print(f"\n[TTA] Running {n}-pass Test Time Augmentation...")
    model.eval()

    tta_ds     = datasets.ImageFolder(val_dir, transform=tta_transforms)
    tta_loader = DataLoader(
        tta_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    all_labels = []
    for _, labels in tta_loader:
        all_labels.extend(labels.numpy())

    num_samples = len(all_labels)
    all_probs   = np.zeros((num_samples, len(classes)))

    for pass_i in range(n):
        start = 0
        for imgs, _ in tta_loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast("cuda"):
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs[start:start + len(probs)] += probs
            start += len(probs)
        print(f"  TTA pass {pass_i+1}/{n} done", end="\r")

    all_probs  /= n
    all_preds   = all_probs.argmax(axis=1)
    all_labels  = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[TTA] Accuracy: {acc*100:.2f}%")
    return acc * 100, all_preds, all_labels


# ═══════════════════════════════════════════
# 7. MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════

def train():
    train_loader, val_loader, classes, class_weights, val_dir = \
        load_datasets(CONFIG["data_dir"])
    num_classes = len(classes)

    model = EmotionNet(num_classes=num_classes).to(device)
    for param in model.parameters():
        param.requires_grad = True

    criterion      = FocalLoss(weight=class_weights,
                               gamma=CONFIG["focal_gamma"],
                               label_smoothing=CONFIG["label_smooth"])
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW([
        {"params": model.backbone.features.parameters(),   "lr": CONFIG["lr_backbone"]},
        {"params": model.backbone.classifier.parameters(), "lr": CONFIG["lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    # ── OneCycleLR only covers epochs 1 → swa_start ──
    onecycle_steps = CONFIG["swa_start"] * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[CONFIG["lr_backbone"] * 10, CONFIG["lr_head"]],
        total_steps=onecycle_steps,      # ← FIXED: exactly swa_start epochs
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )

    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=CONFIG["swa_lr"])
    scaler        = GradScaler("cuda")

    # ── Resume from checkpoint ──
    start_epoch      = 1
    best_val_acc     = 0.0
    patience_counter = 0

    resume_path = CONFIG.get("resume")
    if resume_path and os.path.exists(resume_path):
        print(f"\n[RESUME] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch  = ckpt.get("epoch", 0) + 1
        print(f"[RESUME] Resuming from epoch {start_epoch}  |  Best val acc so far: {best_val_acc:.2f}%")

        # If resuming past swa_start, fast-forward scheduler
        if start_epoch > CONFIG["swa_start"]:
            print(f"[RESUME] Already past SWA start — skipping OneCycleLR fast-forward")
        else:
            # Step scheduler to catch up
            steps_done = (start_epoch - 1) * len(train_loader)
            for _ in range(steps_done):
                scheduler.step()
            print(f"[RESUME] Scheduler fast-forwarded {steps_done} steps")
    else:
        print(f"\n[INFO] Training from scratch")

    print(f"\n[MODEL] EfficientNet-B4 (fully fine-tuned)")
    print(f"[MODEL] Image size  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"[MODEL] Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[MODEL] OneCycleLR  : epochs 1–{CONFIG['swa_start']}  |  SWA: epochs {CONFIG['swa_start']+1}–{CONFIG['num_epochs']}")

    PATIENCE = 15
    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n" + "═" * 72)
    print("  TRAINING  —  EfficientNet-B4  —  Focal Loss  —  SWA")
    print("═" * 72)

    for epoch in range(start_epoch, CONFIG["num_epochs"] + 1):
        t0         = time.time()
        swa_active = epoch > CONFIG["swa_start"]   # SWA starts AFTER swa_start epoch

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, swa_active
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, eval_criterion)
        elapsed = time.time() - t0

        # After swa_start, update SWA model and use SWA LR
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        flag     = " ◀ BEST" if improved else ""
        swa_tag  = " [SWA]" if swa_active else ""

        print(f"Ep [{epoch:3d}/{CONFIG['num_epochs']}]  "
              f"Train {train_loss:.4f}/{train_acc:.1f}%  "
              f"Val {val_loss:.4f}/{val_acc:.2f}%  "
              f"{elapsed:.0f}s{swa_tag}{flag}")

        if improved:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":         best_val_acc,
                "classes":         classes,
                "img_size":        IMG_SIZE,
            }, CONFIG["model_save"])
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and not swa_active:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs.")
                break

    # ── SWA: update BatchNorm stats ──
    print("\n[SWA] Updating BatchNorm statistics (one pass over train data)...")
    update_bn(train_loader, swa_model, device=device)
    torch.save({
        "model_state": swa_model.module.state_dict(),
        "classes":     classes,
        "img_size":    IMG_SIZE,
        "val_acc":     best_val_acc,
    }, CONFIG["swa_save"])
    print(f"[SWA] Saved: {CONFIG['swa_save']}")

    # ── Final evaluation: best checkpoint vs SWA ──
    print("\n" + "═" * 72)
    print("  FINAL EVALUATION")
    print("═" * 72)

    ckpt = torch.load(CONFIG["model_save"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, std_acc, _, _ = evaluate(model, val_loader, eval_criterion)
    tta_acc, tta_preds, tta_labels = evaluate_with_tta(
        model, val_dir, eval_criterion, classes, n=CONFIG["tta_steps"]
    )

    swa_ckpt = torch.load(CONFIG["swa_save"], map_location=device)
    swa_eval = EmotionNet(num_classes=num_classes).to(device)
    swa_eval.load_state_dict(swa_ckpt["model_state"])
    _, swa_acc, _, _ = evaluate(swa_eval, val_loader, eval_criterion)
    swa_tta_acc, swa_tta_preds, swa_tta_labels = evaluate_with_tta(
        swa_eval, val_dir, eval_criterion, classes, n=CONFIG["tta_steps"]
    )

    print(f"\n  Best Checkpoint — Standard: {std_acc:.2f}%  |  TTA: {tta_acc:.2f}%")
    print(f"  SWA Model       — Standard: {swa_acc:.2f}%  |  TTA: {swa_tta_acc:.2f}%")

    if swa_tta_acc >= tta_acc:
        final_preds, final_labels, final_acc = swa_tta_preds, swa_tta_labels, swa_tta_acc
        print(f"\n  → Using SWA model")
    else:
        final_preds, final_labels, final_acc = tta_preds, tta_labels, tta_acc
        print(f"\n  → Using best checkpoint")

    print("\n" + "═" * 72)
    print(f"  FINAL ACCURACY: {final_acc:.2f}%")
    print("═" * 72)
    print(classification_report(final_labels, final_preds,
                                target_names=classes, digits=4))

    metrics = {
        "accuracy":  accuracy_score(final_labels, final_preds),
        "precision": precision_score(final_labels, final_preds,
                                     average="weighted", zero_division=0),
        "recall":    recall_score(final_labels, final_preds,
                                  average="weighted", zero_division=0),
        "f1":        f1_score(final_labels, final_preds,
                              average="weighted", zero_division=0),
    }
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")

    plot_training_history(history)
    plot_confusion_matrix(final_labels, final_preds, classes)
    plot_per_class_accuracy(final_labels, final_preds, classes)

    return model, history, metrics


# ═══════════════════════════════════════════
# 8. PLOTS
# ═══════════════════════════════════════════

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", lw=1.5, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-", lw=1.5, label="Val Loss")
    ax1.axvline(x=CONFIG["swa_start"], color="purple", linestyle="--",
                alpha=0.6, label="SWA Start")
    ax1.set_title("Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", lw=1.5, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-", lw=1.5, label="Val Acc")
    best = max(history["val_acc"])
    ax2.axhline(y=best, color="green", linestyle="--",
                alpha=0.7, label=f"Best: {best:.2f}%")
    ax2.axvline(x=CONFIG["swa_start"], color="purple", linestyle="--",
                alpha=0.6, label="SWA Start")
    ax2.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("EfficientNet-B4 + SWA — Emotion Detection", fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] training_history.png")


def plot_confusion_matrix(labels, preds, classes):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm,      annot=True, fmt="d",   cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] confusion_matrix.png")


def plot_per_class_accuracy(labels, preds, classes):
    cm  = confusion_matrix(labels, preds)
    acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = ["#e74c3c" if a < 60 else "#f39c12" if a < 75 else "#2ecc71"
              for a in acc]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, acc, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(acc), color="navy", linestyle="--",
               alpha=0.7, label=f"Mean: {np.mean(acc):.1f}%")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] per_class_accuracy.png")


if __name__ == "__main__":
    train()