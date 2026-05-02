# Emotion Detection

A Deep Learning-based computer vision project to detect human emotions in real-time. This project uses a highly optimized **EfficientNet-B4** backbone trained in PyTorch, leveraging state-of-the-art training techniques like Stochastic Weight Averaging (SWA), Focal Loss, Mixup/Cutmix, and Test-Time Augmentation (TTA) to achieve maximum accuracy.

## 🌟 Features

- **7-Class Emotion Recognition:** Detects `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
- **High-Performance Backbone:** Utilizes `EfficientNet-B4` modified with a custom classifier head.
- **Advanced Training Pipeline:**
  - **Focal Loss:** Handles class imbalances.
  - **Data Augmentation:** Random Erasing, TTA (Test Time Augmentation), Mixup, and Cutmix.
  - **Optimization:** OneCycleLR learning rate scheduling coupled with SWA (Stochastic Weight Averaging) for better generalization.
- **Real-Time Inference:** Includes a live webcam inference script (`webcam_inference.py`) using OpenCV.
- **Model Comparison:** Utilities to benchmark the model against other ML models (`compare_models.py`).
- **Comprehensive Visualizations:** Automatic generation of training histories, confusion matrices, and per-class accuracy plots.

## 📁 Project Structure

- `train_model.py`: Main script for training the EfficientNet-B4 model with SWA and OneCycleLR.
- `webcam_inference.py`: Run real-time emotion detection using your webcam.
- `compare_models.py`: Script to compare the performance of different model architectures.
- `generate_graphs.py`: Utility to generate evaluation metrics and visualizations.
- `back.py`: Backend integration script.
- `data/`: Directory containing the training and validation image datasets.

## ⚙️ Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repository-url>
   cd Emotion-Detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv311
   # On Windows
   venv311\Scripts\activate
   # On macOS/Linux
   source venv311/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note for GPU Users:** If you have an NVIDIA GPU (e.g., RTX series), ensure you install the CUDA-enabled version of PyTorch for significantly faster training:
   > `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

## 🚀 Usage

### 1. Training the Model
Ensure your data is placed in the `data/train` and `data/validation` directories, then run:
```bash
python train_model.py
```
This will output `best_model.pth` and `swa_model.pth`, as well as performance graphs (`training_history.png`, `confusion_matrix.png`, etc.).

### 2. Live Webcam Inference
To test the model in real-time using your computer's webcam:
```bash
python webcam_inference.py
```
Press `q` to exit the webcam window.

### 3. Comparing Models
If you want to run the model comparison script:
```bash
python compare_models.py
```

## 📊 Metrics & Evaluation
During training, the system automatically evaluates using Test-Time Augmentation (TTA) and outputs detailed metrics including:
- Accuracy, Precision, Recall, and F1-Score
- Normalized Confusion Matrix
- Per-class accuracy bar charts

## 📄 License
This project is open-source and available under standard MIT guidelines.