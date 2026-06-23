# Dog Breed Classifier using Transfer Learning (ResNet50)

A deep learning image classification project that identifies **10 dog breeds** from photos using transfer learning on a pretrained ResNet50 model, built with PyTorch.

---

## Overview

This project fine-tunes a ResNet50 model (pretrained on ImageNet) to classify dog images into one of 10 breeds. It achieves strong accuracy on a relatively small dataset (~967 images) by leveraging transfer learning and data augmentation techniques.

---

## Dog Breeds Supported

| # | Breed |
|---|-------|
| 1 | Beagle |
| 2 | Boxer |
| 3 | Bulldog |
| 4 | Dachshund |
| 5 | German Shepherd |
| 6 | Golden Retriever |
| 7 | Labrador Retriever |
| 8 | Poodle |
| 9 | Rottweiler |
| 10 | Yorkshire Terrier |

---

## Project Structure

```
Dog Classifier/
│
├── dataset/                        # Image dataset (10 breed folders, ~100 images each)
│   ├── Beagle/
│   ├── Boxer/
│   ├── Bulldog/
│   ├── Dachshund/
│   ├── German_Shepherd/
│   ├── Golden_Retriever/
│   ├── Labrador_Retriever/
│   ├── Poodle/
│   ├── Rottweiler/
│   └── Yorkshire_Terrier/
│
├── output/                         # Generated after training
│   ├── dog_classifier.pth          # Saved model weights
│   ├── training_curves.png         # Loss & accuracy plots
│   └── confusion_matrix.png        # Evaluation confusion matrix
│
├── dog_breed_classification.ipynb  # Full walkthrough notebook
├── train.py                        # Standalone training script
├── predict.py                      # Predict breed from a new image
└── README.md
```

---

## Model Architecture

- **Base Model:** ResNet50 pretrained on ImageNet (`IMAGENET1K_V2` weights)
- **Fine-tuning Strategy:** All early layers frozen; only `layer4` + classifier unfrozen
- **Custom Classification Head:**
  ```
  Dropout(0.4) → Linear(2048 → 256) → ReLU → Dropout(0.2) → Linear(256 → 10)
  ```
- **Total Parameters:** ~25.6M (only ~6.8M trainable)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss Function | CrossEntropyLoss |
| Train/Val Split | 80% / 20% |
| Random Seed | 42 |

---

## Data Augmentation

Applied to training images to reduce overfitting on the small dataset:

- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Random Affine Translation (±10%)
- ImageNet Normalization

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dog-classifier.git
cd dog-classifier
```

### 2. Install dependencies
```bash
pip install torch torchvision scikit-learn matplotlib pillow
```

---

## How to Run

### Option 1 — Jupyter Notebook (Recommended)
Open and run `dog_breed_classification.ipynb` step by step. It includes:
- Dataset exploration and visualization
- Augmented sample previews
- Training with live progress
- Training curves and confusion matrix
- Single-image prediction with confidence scores

### Option 2 — Command Line

**Train the model:**
```bash
python train.py
```

**Predict a dog breed from an image:**
```bash
python predict.py path/to/your/dog_photo.jpg
python predict.py path/to/your/dog_photo.jpg --top 5
```

---

## Output

After training, the following files are saved to the `output/` folder:

- `dog_classifier.pth` — Saved model weights and class names
- `training_curves.png` — Train vs validation loss and accuracy over 20 epochs
- `confusion_matrix.png` — Per-class prediction accuracy heatmap

### Example Prediction Output
```
Predictions for: my_dog.jpg

  1. Golden Retriever          87.3%  █████████████████████████████████████████
  2. Labrador Retriever         9.1%  ████
  3. Beagle                     2.3%  █
```

---

## Technologies Used

- **Python 3.x**
- **PyTorch** — model building and training
- **Torchvision** — pretrained ResNet50 and image transforms
- **Scikit-learn** — classification report and confusion matrix
- **Matplotlib** — visualizations
- **Pillow** — image loading

---

## Key Concepts Demonstrated

- **Transfer Learning** — reusing pretrained ImageNet weights for a new task
- **Fine-tuning** — selectively unfreezing deep layers for domain adaptation
- **Data Augmentation** — improving generalization with a small dataset
- **Learning Rate Scheduling** — adaptive LR reduction on plateau
- **Model Checkpointing** — saving the best model based on validation accuracy

---

## Author

**Amoako Emmanuel Kwame**
MSc Data Science, AI and Digital Business
Gisma University of Applied Sciences, Germany
