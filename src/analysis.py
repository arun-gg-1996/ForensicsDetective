import cv2
import numpy as np
import os
import sys
import glob
import torch
import torch.nn as nn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# add src folder to path so we can import the CNN class
sys.path.append(os.path.join(os.path.dirname(__file__)))
from cnn_classifier import SimpleCNN


# --- device setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- preprocessing functions ---

def preprocess_for_xgboost(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to 64x64
    resized = cv2.resize(gray, (64, 64))
    # flatten and normalize
    flat = resized.flatten() / 255.0
    return flat


def preprocess_for_cnn(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize to 128x128
    resized = cv2.resize(gray, (128, 128))
    # normalize and convert to tensor
    tensor = torch.FloatTensor(resized / 255.0)
    # add channel dimension (1, 128, 128)
    tensor = tensor.unsqueeze(0)
    return tensor


# --- load images from an augmentation folder ---

def load_augmented_images(aug_folder):
    # each augmentation folder has 3 subfolders
    subfolders = [
        ("word_pdfs_png", 0),
        ("google_docs_pdfs_png", 1),
        ("python_pdfs_png", 2),
    ]

    images = []
    labels = []

    for subfolder, label in subfolders:
        folder_path = os.path.join(aug_folder, subfolder)
        files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
                labels.append(label)

    return images, labels


# --- evaluate a model on a set of images ---

def evaluate_xgboost(model, images, labels):
    # preprocess all images for xgboost
    X = np.array([preprocess_for_xgboost(img) for img in images])
    y = np.array(labels)

    # predict
    y_pred = model.predict(X)

    # compute metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    return y, y_pred, acc, prec, rec, f1


def evaluate_cnn(model, images, labels):
    # preprocess all images for cnn
    tensors = [preprocess_for_cnn(img) for img in images]

    # predict in batches
    model.eval()
    all_preds = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size]).to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    y = np.array(labels)
    y_pred = np.array(all_preds)

    # compute metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    return y, y_pred, acc, prec, rec, f1


# --- save confusion matrix plot ---

def save_confusion_matrix(y_true, y_pred, title, save_path):
    class_names = ["Word (0)", "Google Docs (1)", "Python (2)"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- main script ---

def main():
    print("=" * 60)
    print("Robustness Analysis - XGBoost & CNN vs Augmented Images")
    print("=" * 60)

    # create output directories
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/robustness_plots", exist_ok=True)

    # --- step 1: load both trained models ---
    print("\nStep 1: Loading trained models...")

    # load xgboost
    xgb_model = joblib.load("results/xgboost_model.joblib")
    print("  Loaded XGBoost model from results/xgboost_model.joblib")

    # load cnn
    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load("results/cnn_model.pth", map_location=device))
    cnn_model.eval()
    print("  Loaded CNN model from results/cnn_model.pth")

    # --- step 2: define augmentation types to test ---
    augmentation_types = ["original", "gaussian", "jpeg", "dpi", "crop", "bitdepth"]
    base_path = "data/augmented_images"

    # store all results here
    all_results = []

    # --- step 3: evaluate both models on each augmentation ---
    print("\nStep 3: Evaluating models on each augmentation type...\n")

    xgb_accuracies = []
    cnn_accuracies = []

    for aug_type in augmentation_types:
        aug_folder = os.path.join(base_path, aug_type)
        print(f"--- Testing on: {aug_type} ---")

        # load images for this augmentation
        images, labels = load_augmented_images(aug_folder)
        print(f"  Loaded {len(images)} images")

        # evaluate xgboost
        print(f"  Evaluating XGBoost...")
        y_true_xgb, y_pred_xgb, xgb_acc, xgb_prec, xgb_rec, xgb_f1 = \
            evaluate_xgboost(xgb_model, images, labels)

        print(f"    Accuracy: {xgb_acc:.4f}  Precision: {xgb_prec:.4f}  "
              f"Recall: {xgb_rec:.4f}  F1: {xgb_f1:.4f}")

        xgb_accuracies.append(xgb_acc)

        # save xgboost confusion matrix
        cm_path = f"results/confusion_matrices/xgboost_{aug_type}.png"
        save_confusion_matrix(y_true_xgb, y_pred_xgb,
                              f"XGBoost - {aug_type}", cm_path)

        # store results
        all_results.append({
            "model": "XGBoost",
            "augmentation_type": aug_type,
            "accuracy": round(xgb_acc, 4),
            "precision": round(xgb_prec, 4),
            "recall": round(xgb_rec, 4),
            "f1": round(xgb_f1, 4),
        })

        # evaluate cnn
        print(f"  Evaluating CNN...")
        y_true_cnn, y_pred_cnn, cnn_acc, cnn_prec, cnn_rec, cnn_f1 = \
            evaluate_cnn(cnn_model, images, labels)

        print(f"    Accuracy: {cnn_acc:.4f}  Precision: {cnn_prec:.4f}  "
              f"Recall: {cnn_rec:.4f}  F1: {cnn_f1:.4f}")

        cnn_accuracies.append(cnn_acc)

        # save cnn confusion matrix
        cm_path = f"results/confusion_matrices/cnn_{aug_type}.png"
        save_confusion_matrix(y_true_cnn, y_pred_cnn,
                              f"CNN - {aug_type}", cm_path)

        # store results
        all_results.append({
            "model": "CNN",
            "augmentation_type": aug_type,
            "accuracy": round(cnn_acc, 4),
            "precision": round(cnn_prec, 4),
            "recall": round(cnn_rec, 4),
            "f1": round(cnn_f1, 4),
        })

        print()

    # --- step 4: save all metrics to csv ---
    print("Step 4: Saving performance metrics to CSV...")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/performance_metrics.csv", index=False)
    print("  Saved to results/performance_metrics.csv")
    print()
    print(results_df.to_string(index=False))

    # --- step 5: generate robustness curve plot ---
    print("\n\nStep 5: Generating robustness curve plot...")

    plt.figure(figsize=(10, 6))
    plt.plot(augmentation_types, xgb_accuracies, 'o-', label='XGBoost', linewidth=2, markersize=8)
    plt.plot(augmentation_types, cnn_accuracies, 's-', label='CNN', linewidth=2, markersize=8)
    plt.xlabel("Augmentation Type")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve: Accuracy vs Augmentation Type")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("results/robustness_plots/robustness_curve.png", dpi=150)
    plt.close()
    print("  Saved to results/robustness_plots/robustness_curve.png")

    # --- step 6: print summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # find biggest accuracy drop for xgboost
    xgb_baseline = xgb_accuracies[0]
    xgb_drops = [(aug, xgb_baseline - acc) for aug, acc in
                 zip(augmentation_types[1:], xgb_accuracies[1:])]
    xgb_worst = max(xgb_drops, key=lambda x: x[1])

    print(f"\nXGBoost baseline accuracy (original): {xgb_baseline:.4f}")
    print(f"  Biggest drop: {xgb_worst[0]} (accuracy dropped by {xgb_worst[1]:.4f})")

    # find biggest accuracy drop for cnn
    cnn_baseline = cnn_accuracies[0]
    cnn_drops = [(aug, cnn_baseline - acc) for aug, acc in
                 zip(augmentation_types[1:], cnn_accuracies[1:])]
    cnn_worst = max(cnn_drops, key=lambda x: x[1])

    print(f"\nCNN baseline accuracy (original): {cnn_baseline:.4f}")
    print(f"  Biggest drop: {cnn_worst[0]} (accuracy dropped by {cnn_worst[1]:.4f})")

    # overall comparison
    print(f"\nAccuracy comparison across all augmentations:")
    for i, aug in enumerate(augmentation_types):
        print(f"  {aug:12s}  XGBoost: {xgb_accuracies[i]:.4f}  CNN: {cnn_accuracies[i]:.4f}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
