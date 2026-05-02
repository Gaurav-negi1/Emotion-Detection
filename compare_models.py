"""
============================================================
Task 3-7: Comparative Model Analysis
Implements 4 classifiers on FER feature vectors extracted
from pretrained CNN backbone — satisfies professor requirement
of comparing 4 ML models.

Models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Support Vector Machine (SVM)

Run AFTER train_model.py (needs best_model.pth)
Output: comparative_analysis.png + comparison_table.csv
============================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ─── Config ───────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_PATH  = "best_model.pth"
IMG_SIZE    = 96
BATCH_SIZE  = 128
NUM_WORKERS = 4
EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ─── Step 1: Extract deep features using trained backbone ──

class FeatureExtractor(nn.Module):
    """Strip the classifier head → 1408-dim embedding from EfficientNet-B2"""
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        backbone = models.efficientnet_b2(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


def extract_features(loader, model):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)


def get_features():
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),   transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    extractor = FeatureExtractor(num_classes=7).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    # Load only the backbone weights
    state = {k.replace("backbone.", "", 1): v
             for k, v in ckpt["model_state"].items() if "backbone." in k}
    # Load features + avgpool from backbone
    feat_state = {k: v for k, v in state.items()
                  if k.startswith("features.") or k.startswith("avgpool.")}
    extractor.load_state_dict(feat_state, strict=False)
    extractor.eval()

    print("[FEATURES] Extracting training features...")
    X_train, y_train = extract_features(train_loader, extractor)
    print("[FEATURES] Extracting validation features...")
    X_val,   y_val   = extract_features(val_loader,   extractor)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    print(f"[FEATURES] Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ─── Step 2: Train & evaluate 4 classifiers ───────────────

def run_classifier(name, clf, X_train, y_train, X_val, y_val):
    print(f"\n[{name}] Training...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = clf.predict(X_val)
    acc   = accuracy_score(y_val, preds)
    prec  = precision_score(y_val, preds, average="weighted", zero_division=0)
    rec   = recall_score(y_val, preds, average="weighted", zero_division=0)
    f1    = f1_score(y_val, preds, average="weighted", zero_division=0)

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Time      : {train_time:.2f}s")

    return {
        "Model":         name,
        "Accuracy (%)":  round(acc * 100, 2),
        "Precision":     round(prec, 4),
        "Recall":        round(rec, 4),
        "F1-Score":      round(f1, 4),
        "Training Time": f"{train_time:.2f}s",
        "preds":         preds,
    }


# ─── Step 3: Plot all results ──────────────────────────────

def plot_comparison(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparative Model Analysis — Emotion Detection", fontsize=16, fontweight="bold", y=1.01)

    metrics = ["Accuracy (%)", "Precision", "Recall", "F1-Score"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, metric in zip(axes.flatten(), metrics):
        bars = ax.bar(results_df["Model"], results_df[metric], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_ylim(0, results_df[metric].max() * 1.15)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("comparative_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved: comparative_analysis.png")


def plot_confusion_matrices(results, y_val):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

    for ax, r in zip(axes.flatten(), results):
        cm = confusion_matrix(y_val, r["preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax,
                    linewidths=0.5)
        ax.set_title(r["Model"], fontweight="bold")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("all_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PLOT] Saved: all_confusion_matrices.png")


# ─── Main ─────────────────────────────────────────────────

def main():
    X_train, y_train, X_val, y_val = get_features()

    classifiers = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=20, min_samples_split=10,
                                random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, max_depth=25,
                                min_samples_split=5, n_jobs=-1, random_state=42)),
        ("SVM (RBF)",
         SVC(C=10.0, gamma="scale", kernel="rbf", probability=True)),
    ]

    results = []
    for name, clf in classifiers:
        r = run_classifier(name, clf, X_train, y_train, X_val, y_val)
        results.append(r)

    # Build comparison table
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in results])
    print("\n" + "═" * 65)
    print("  COMPARATIVE SUMMARY TABLE")
    print("═" * 65)
    print(df.to_string(index=False))
    df.to_csv("comparison_table.csv", index=False)
    print("\n[CSV] Saved: comparison_table.csv")

    # Identify best
    best_row = df.loc[df["Accuracy (%)"].idxmax()]
    print(f"\n[RESULT] Best Model: {best_row['Model']} "
          f"({best_row['Accuracy (%)']:.2f}% accuracy)")

    plot_comparison(df)
    plot_confusion_matrices(results, y_val)

    return df


if __name__ == "__main__":
    main()