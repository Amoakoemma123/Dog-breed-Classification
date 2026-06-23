"""
Dog Breed Classification - Training Script
10 breeds: Beagle, Boxer, Bulldog, Dachshund, German Shepherd,
           Golden Retriever, Labrador Retriever, Poodle, Rottweiler, Yorkshire Terrier
Uses transfer learning with ResNet50 pretrained on ImageNet.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.8  # 80% train, 20% val
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Data Transforms ────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Load & Split Dataset ───────────────────────────────────────────────────
def create_dataloaders():
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # Split into train/val
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(dataset_size * TRAIN_SPLIT)
    train_indices, val_indices = indices[:split], indices[split:]

    # Create subsets with different transforms
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Device: {DEVICE}")
    return train_loader, val_loader, class_names, num_classes


# ─── Build Model ─────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers, only train later layers + classifier
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the classification head
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    return model.to(DEVICE)


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      factor=0.5, patience=3)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # ── Train phase ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validation phase ──
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        scheduler.step(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, val_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n── Classification Report ──")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return np.array(all_labels), np.array(all_preds)


# ─── Plots ───────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Saved training curves to {OUTPUT_DIR}/training_curves.png")


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    short_names = [n.replace("_", "\n") for n in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=8)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Dog Breed Classification - Transfer Learning (ResNet50)")
    print("=" * 60)

    train_loader, val_loader, class_names, num_classes = create_dataloaders()
    model = build_model(num_classes)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total\n")

    model, history = train_model(model, train_loader, val_loader)

    # Evaluate
    labels, preds = evaluate(model, val_loader, class_names)

    # Save plots
    plot_history(history)
    plot_confusion_matrix(labels, preds, class_names)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "dog_classifier.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
    }, model_path)
    print(f"Saved model to {model_path}")
