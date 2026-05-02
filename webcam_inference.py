"""
============================================================
Emotion Detection - Real-Time Webcam Inference
Updated for: EfficientNet-B4 (matches train_model.py)
Uses: swa_model.pth (best accuracy) or best_model.pth
Press Q to quit
============================================================
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from collections import deque
import time

# ═══════════════════════════════════════════
# MODEL — must exactly match train_model.py
# EfficientNet-B4, 3-layer head
# ═══════════════════════════════════════════

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
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
# CONFIG
# ═══════════════════════════════════════════

# Use SWA model if available (slightly higher accuracy), else best checkpoint
import os
MODEL_PATH = "swa_model.pth" if os.path.exists("swa_model.pth") else "best_model.pth"

SMOOTHING_WINDOW = 10   # frames to average predictions over

EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (150, 0,   150),
    "happy":    (0,   200, 200),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 0),
    "surprise": (0,   200, 0),
}

# ═══════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device    : {device}")
if device.type == "cuda":
    print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}")
print(f"[INFO] Loading   : {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes    = checkpoint["classes"]
img_size   = checkpoint.get("img_size", 224)

model = EmotionNet(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"[INFO] Classes   : {classes}")
print(f"[INFO] Image size: {img_size}x{img_size}")
if "val_acc" in checkpoint:
    print(f"[INFO] Val Acc   : {checkpoint['val_acc']:.2f}%")

# Inference transform — matches val_transforms in train_model.py
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Haar face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def predict_emotion(face_img):
    x = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = probs.argmax()
    return classes[idx], float(probs[idx]), probs


def draw_prob_bars(frame, probs, classes, x0, y0, w=150, bar_h=13, gap=4):
    for i, (cls, p) in enumerate(zip(classes, probs)):
        color = EMOTION_COLORS.get(cls, (200, 200, 200))
        y = y0 + i * (bar_h + gap)
        bar_w = int(p * w)
        cv2.rectangle(frame, (x0, y), (x0 + w, y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), color, -1)
        label = f"{cls[:3].upper()} {p*100:.0f}%"
        cv2.putText(frame, label, (x0 + w + 5, y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════
# MAIN WEBCAM LOOP
# ═══════════════════════════════════════════

def run_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    smooth_bufs = [deque(maxlen=SMOOTHING_WINDOW) for _ in range(4)]
    fps_buf     = deque(maxlen=30)
    prev_t      = time.time()

    print("\n[WEBCAM] Running... Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        t = time.time()
        fps_buf.append(1.0 / max(t - prev_t, 1e-6))
        prev_t = t
        fps    = np.mean(fps_buf)

        # Face detection
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for fi, (x, y, w, h) in enumerate(faces[:4]):
            pad = int(0.12 * min(w, h))
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            emotion, conf, probs = predict_emotion(face_crop)

            # Smooth
            smooth_bufs[fi].append(probs)
            avg_probs     = np.mean(smooth_bufs[fi], axis=0)
            smooth_idx    = avg_probs.argmax()
            smooth_emotion = classes[smooth_idx]
            smooth_conf   = float(avg_probs[smooth_idx])

            color = EMOTION_COLORS.get(smooth_emotion, (255, 255, 255))

            # Face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label above face
            label = f"{smooth_emotion.upper()}  {smooth_conf*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Probability bars
            if x2 + 180 < frame.shape[1]:
                draw_prob_bars(frame, avg_probs, classes, x2 + 10, y1)

        # HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 55), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        cv2.putText(frame, f"Emotion Detection  |  FPS: {fps:.1f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {len(faces)}  |  EfficientNet-B4  |  Q=Quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")


if __name__ == "__main__":
    run_webcam()