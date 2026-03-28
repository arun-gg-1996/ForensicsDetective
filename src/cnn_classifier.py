import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# check which device to use
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- dataset class ---

class PDFDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load the image
        img = cv2.imread(self.image_paths[idx])

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize to 128x128
        resized = cv2.resize(gray, (128, 128))

        # normalize to [0, 1] and convert to tensor
        tensor = torch.FloatTensor(resized / 255.0)

        # add channel dimension (1, 128, 128)
        tensor = tensor.unsqueeze(0)

        label = self.labels[idx]
        return tensor, label


# --- cnn model ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # first conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # fully connected layers
        # after 3 maxpools: 128 -> 64 -> 32 -> 16, so 128 * 16 * 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


# --- helper function to predict a single image ---

def predict_image(image_path, model, device):
    # load and preprocess one image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))

    # convert to tensor and normalize
    tensor = torch.FloatTensor(resized / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # add batch and channel dims

    # move to device and predict
    tensor = tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # map class number to name
    class_names = ["Word", "Google Docs", "Python"]
    pred_class = class_names[predicted.item()]
    conf_pct = confidence.item() * 100

    return pred_class, conf_pct


# --- main script ---

def main():
    print("=" * 50)
    print("CNN Classifier - Training on Original Images")
    print("=" * 50)

    # create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # --- step 1: load all image paths and labels ---
    print("\nStep 1: Loading image paths...")

    folders = [
        ("word_pdfs_png", 0),
        ("google_docs_pdfs_png", 1),
        ("python_pdfs_png", 2),
    ]

    all_paths = []
    all_labels = []

    for folder, label in folders:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        print(f"  Found {len(files)} images in {folder} (label={label})")
        for f in files:
            all_paths.append(f)
            all_labels.append(label)

    print(f"Total images: {len(all_paths)}")

    # --- step 2: split into train and test ---
    print("\nStep 2: Splitting into train/test (80/20)...")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"Training set: {len(train_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    # --- step 3: create datasets and dataloaders ---
    print("\nStep 3: Creating datasets and dataloaders...")

    train_dataset = PDFDataset(train_paths, train_labels)
    test_dataset = PDFDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # --- step 4: create model, optimizer, loss function ---
    print("\nStep 4: Setting up model...")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Model moved to {device}")

    # --- step 5: train the model ---
    print("\nStep 5: Training for 15 epochs...")

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # move batch to device
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # print epoch results
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/15 - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.1f}%")

    print("Training complete!")

    # --- step 6: evaluate on test set ---
    print("\nStep 6: Evaluating on test set...")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            # move to device
            images = images.to(device)

            # get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # move back to cpu for sklearn
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())

    # convert to numpy arrays
    y_test = np.array(all_true)
    y_pred = np.array(all_preds)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # print full classification report
    class_names = ["Word (0)", "Google Docs (1)", "Python (2)"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- step 7: save confusion matrix plot ---
    print("Step 7: Saving confusion matrix plot...")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices/cnn_confusion.png", dpi=150)
    plt.close()
    print("  Saved to results/confusion_matrices/cnn_confusion.png")

    # --- step 8: save the trained model ---
    print("\nStep 8: Saving trained model...")
    torch.save(model.state_dict(), "results/cnn_model.pth")
    print("  Saved to results/cnn_model.pth")

    # --- step 9: save test results to csv ---
    print("\nStep 9: Saving test predictions to CSV...")
    results_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
    })
    results_df.to_csv("results/cnn_test_results.csv", index=False)
    print("  Saved to results/cnn_test_results.csv")

    # --- quick demo of predict_image ---
    print("\n--- Quick Demo ---")
    sample_image = all_paths[0]
    pred_class, conf = predict_image(sample_image, model, device)
    print(f"Image: {sample_image}")
    print(f"Predicted: {pred_class} ({conf:.1f}% confidence)")

    print("\nAll done!")


if __name__ == "__main__":
    main()
