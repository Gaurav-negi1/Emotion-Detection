"""
============================================================
Emotion Detection - Flask Backend (Web Inference)
Optimized for low latency + auto-detects B2/B4 architecture
============================================================
"""

# ── Thread tuning MUST come before torch import ──
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import io
import time
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms, models

torch.set_num_threads(4)

# ═══════════════════════════════════════════
# 1. MODEL DEFINITIONS (B2 and B4)
# ═══════════════════════════════════════════

class EmotionNetB2(nn.Module):
    """Matches back.py — EfficientNet-B2, img_size=96"""
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1408
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


class EmotionNetB4(nn.Module):
    """Matches train_model.py — EfficientNet-B4, img_size=224"""
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
# 2. LOAD MODEL — auto-detect architecture
# ═══════════════════════════════════════════

app = Flask(__name__, static_folder='.', static_url_path='')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Prefer B2 (faster on CPU). Comment this out if you want B4.
# MODEL_PATH = "best_model.pth" if os.path.exists("best_model.pth") else "swa_model.pth"
MODEL_PATH = "swa_model.pth" if os.path.exists("swa_model.pth") else "best_model.pth"
print(f"[INFO] Loading model from: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes    = checkpoint["classes"]
img_size   = checkpoint.get("img_size", 224)

# Detect architecture from the checkpoint's first conv layer
state_dict = checkpoint["model_state"]
first_conv_key = [k for k in state_dict.keys() if "features.0.0.weight" in k][0]
out_channels = state_dict[first_conv_key].shape[0]

if out_channels == 32:
    print(f"[INFO] Detected EfficientNet-B2 architecture (img_size={img_size})")
    model = EmotionNetB2(num_classes=len(classes)).to(device)
elif out_channels == 48:
    print(f"[INFO] Detected EfficientNet-B4 architecture (img_size={img_size})")
    model = EmotionNetB4(num_classes=len(classes)).to(device)
else:
    raise ValueError(f"Unknown architecture — first conv has {out_channels} channels")

model.load_state_dict(state_dict)
model.eval()
print(f"[INFO] Model loaded. Classes: {classes}")
if "val_acc" in checkpoint:
    print(f"[INFO] Checkpoint val_acc: {checkpoint['val_acc']:.2f}%")

# Optional: reduce inference resolution for speed
INFERENCE_SIZE = img_size  # e.g. set to 160 for B4 to speed up

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Warm-up inference so the first real request isn't slow
print("[INFO] Warming up model...")
_warm = torch.zeros(1, 3, INFERENCE_SIZE, INFERENCE_SIZE, device=device)
with torch.no_grad():
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        _ = model(_warm)
print("[INFO] Model ready.")


# ═══════════════════════════════════════════
# 3. ROUTES
# ═══════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    t_start = time.perf_counter()
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data  = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image       = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx    = int(probs.argmax())
        pred_class  = classes[pred_idx]
        confidence  = float(probs[pred_idx])
        prob_dict   = {classes[i]: float(probs[i]) for i in range(len(classes))}
        server_ms   = (time.perf_counter() - t_start) * 1000

        return jsonify({
            'emotion':       pred_class,
            'confidence':    confidence,
            'probabilities': prob_dict,
            'server_ms':     server_ms,
            'arch':          'B2' if out_channels == 32 else 'B4',
            'img_size':      INFERENCE_SIZE,
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════
# 4. RUN
# ═══════════════════════════════════════════
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)