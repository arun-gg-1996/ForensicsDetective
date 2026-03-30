import numpy as np
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import joblib

# add src folder to path so I can import my modules
sys.path.append(os.path.dirname(__file__))
from utils import (load_images_from_folders, preprocess_image,
                   get_device, save_confusion_matrix, print_metrics)
from cnn_classifier import SimpleCNN


# --- dataset class for CNN training ---
# works with grayscale images already loaded in memory

class GrayscaleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # resize to 128x128 for CNN
        resized = cv2.resize(self.images[idx], (128, 128))
        # normalize and convert to tensor
        tensor = torch.FloatTensor(resized / 255.0)
        # add channel dimension (1, 128, 128)
        tensor = tensor.unsqueeze(0)
        return tensor, self.labels[idx]


# --- main training pipeline ---

def main():
    print("=" * 60)
    print("Classification Pipeline - Training All 4 Classifiers")
    print("=" * 60)

    # create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # =============================================
    # STEP 1: LOAD IMAGES
    # =============================================
    print("\nStep 1: Loading images...")
    images, labels = load_images_from_folders(".")
    print(f"Class distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}")

    # =============================================
    # STEP 2: PREPROCESS FOR SKLEARN MODELS
    # =============================================
    print("\nStep 2: Preprocessing images for sklearn models (64x64 flattened)...")
    X = np.array([preprocess_image(img) for img in images])
    y = labels
    print(f"Feature matrix shape: {X.shape}")

    # =============================================
    # STEP 3: TRAIN/TEST SPLIT
    # =============================================
    print("\nStep 3: Splitting into train/test (80/20, stratified)...")

    # split using indices so I can reuse for both sklearn and CNN
    indices = np.arange(len(images))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    # sklearn data
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # CNN data (raw grayscale images)
    train_images_cnn = [images[i] for i in idx_train]
    test_images_cnn = [images[i] for i in idx_test]
    y_train_cnn = y[idx_train]
    y_test_cnn = y[idx_test]

    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")

    # store results for the summary table
    results_summary = {}

    # =============================================
    # STEP 4A: TRAIN XGBOOST
    # =============================================
    print("\n" + "=" * 60)
    print("Step 4a: Training XGBoost Classifier...")
    print("  n_estimators=100, max_depth=6, learning_rate=0.1")
    print("=" * 60)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, verbose=True)
    print("XGBoost training complete!")

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = print_metrics(y_test, xgb_pred, "XGBoost")
    save_confusion_matrix(y_test, xgb_pred, "xgboost", "original",
                          "results/confusion_matrices")

    joblib.dump(xgb_model, "results/xgboost_model.joblib")
    print("  Saved model to results/xgboost_model.joblib")
    results_summary["XGBoost"] = xgb_acc

    # =============================================
    # STEP 4B: TRAIN SVM
    # =============================================
    print("\n" + "=" * 60)
    print("Step 4b: Training SVM Classifier...")
    print("  kernel='rbf', C=1.0, gamma='scale'")
    print("=" * 60)

    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
    )
    svm_model.fit(X_train, y_train)
    print("SVM training complete!")

    svm_pred = svm_model.predict(X_test)
    svm_acc = print_metrics(y_test, svm_pred, "SVM")
    save_confusion_matrix(y_test, svm_pred, "svm", "original",
                          "results/confusion_matrices")

    joblib.dump(svm_model, "results/svm_model.joblib")
    print("  Saved model to results/svm_model.joblib")
    results_summary["SVM"] = svm_acc

    # =============================================
    # STEP 4C: TRAIN SGD
    # =============================================
    print("\n" + "=" * 60)
    print("Step 4c: Training SGD Classifier...")
    print("  loss='modified_huber', max_iter=1000")
    print("=" * 60)

    sgd_model = SGDClassifier(
        loss='modified_huber',
        max_iter=1000,
        random_state=42,
    )
    sgd_model.fit(X_train, y_train)
    print("SGD training complete!")

    sgd_pred = sgd_model.predict(X_test)
    sgd_acc = print_metrics(y_test, sgd_pred, "SGD")
    save_confusion_matrix(y_test, sgd_pred, "sgd", "original",
                          "results/confusion_matrices")

    joblib.dump(sgd_model, "results/sgd_model.joblib")
    print("  Saved model to results/sgd_model.joblib")
    results_summary["SGD"] = sgd_acc

    # =============================================
    # STEP 4D: TRAIN CNN
    # =============================================
    print("\n" + "=" * 60)
    print("Step 4d: Training CNN Classifier...")
    print("  128x128 input, 15 epochs, batch_size=32")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    # create datasets and dataloaders
    train_dataset = GrayscaleDataset(train_images_cnn, y_train_cnn)
    test_dataset = GrayscaleDataset(test_images_cnn, y_test_cnn)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # create model, optimizer, loss
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print(f"Model moved to {device}")

    # train for 15 epochs
    print("\nTraining for 15 epochs...")
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"  Epoch {epoch+1}/15 - Loss: {epoch_loss:.4f} - Train Accuracy: {epoch_acc:.1f}%")

    print("CNN training complete!")

    # evaluate CNN on test set
    print("\nEvaluating CNN on test set...")
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(batch_labels.numpy())

    cnn_true = np.array(all_true)
    cnn_pred = np.array(all_preds)

    cnn_acc = print_metrics(cnn_true, cnn_pred, "CNN")
    save_confusion_matrix(cnn_true, cnn_pred, "cnn", "original",
                          "results/confusion_matrices")

    torch.save(model.state_dict(), "results/cnn_model.pth")
    print("  Saved model to results/cnn_model.pth")
    results_summary["CNN"] = cnn_acc

    # =============================================
    # STEP 5: SUMMARY TABLE
    # =============================================
    print("\n" + "=" * 60)
    print("SUMMARY - All 4 Classifiers on Original Images")
    print("=" * 60)
    print(f"\n  {'Model':<15} {'Accuracy':>10}")
    print(f"  {'-'*15} {'-'*10}")
    for model_name, acc in results_summary.items():
        print(f"  {model_name:<15} {acc*100:>9.2f}%")

    best = max(results_summary, key=results_summary.get)
    print(f"\n  Best model: {best} ({results_summary[best]*100:.2f}%)")
    print("\nAll done!")


if __name__ == "__main__":
    main()
