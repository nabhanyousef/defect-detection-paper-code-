
import os
import cv2
import torch
import numpy as np
from glob import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---- Dataset Class ----
class DefectDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.class_map = {}
        for idx, defect_type in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, defect_type)
            if os.path.isdir(class_dir):
                self.class_map[defect_type] = idx
                for img_path in glob(os.path.join(class_dir, "*.jpg")):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # Denoising
        img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

        # Affine Transform
        h, w = img.shape
        center = (w // 2, h // 2)
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Phong Simulation
        norm_img = img.astype(np.float32) / 255.0
        gx = cv2.Sobel(norm_img, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(norm_img, cv2.CV_32F, 0, 1, ksize=5)
        normal = np.dstack((-gx, -gy, np.ones_like(norm_img)))
        normal /= np.clip(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6, None)
        light = np.array([1, -1, 1]) / np.sqrt(3)
        dot = np.clip(np.sum(normal * light, axis=2), 0, 1)
        specular = (dot ** 20) * 0.5
        phong_img = np.clip(norm_img + specular, 0, 1)
        img = (phong_img * 255).astype(np.uint8)

        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label)

# ---- Simple CNN ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---- Train Model ----
def train_model(root_dir, epochs=10):
    dataset = DefectDataset(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4)

    model = SimpleCNN(num_classes=4).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = torch.argmax(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_accuracy = 100 * correct / total
        train_acc.append(train_accuracy)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                predicted = torch.argmax(pred, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_accuracy = 100 * correct / total
        val_acc.append(val_accuracy)

        print(f"Epoch {epoch+1} - Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

    # Plot accuracy
    plt.plot(range(1, epochs+1), val_acc, label='Validation Accuracy', marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("validation_accuracy_graph.png")

if __name__ == "__main__":
    train_model("defect_dataset")  # <-- update path
