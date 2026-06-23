"""
Dog Breed Classification - Prediction Script
Load the trained model and classify a new dog image.

Usage:
    python predict.py path/to/dog_image.jpg
    python predict.py path/to/dog_image.jpg --top 5
"""

import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "dog_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names


def predict(image_path, top_k=3):
    model, class_names = load_model()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probs.topk(top_k)

    print(f"\nPredictions for: {image_path}\n")
    for i in range(top_k):
        breed = class_names[top_indices[i]].replace("_", " ")
        conf = top_probs[i].item() * 100
        bar = "█" * int(conf / 2)
        print(f"  {i+1}. {breed:<25s} {conf:5.1f}%  {bar}")

    # Show image with prediction
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    top_breed = class_names[top_indices[0]].replace("_", " ")
    top_conf = top_probs[0].item() * 100
    ax.set_title(f"{top_breed} ({top_conf:.1f}%)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict dog breed from image")
    parser.add_argument("image", help="Path to dog image")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
    else:
        predict(args.image, args.top)
