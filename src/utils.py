import cv2
import numpy as np
import os
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)


print("utils.py loaded successfully!")


# --- load all images from the 3 class folders ---

def load_images_from_folders(base_path):
    # folders and their labels
    folders = [
        ("word_pdfs_png", 0),       # Word
        ("google_docs_pdfs_png", 1), # Google Docs
        ("python_pdfs_png", 2),      # Python/ReportLab
    ]

    images = []
    labels = []

    for folder_name, label in folders:
        folder_path = os.path.join(base_path, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        print(f"  Found {len(files)} images in {folder_name} (label={label})")

        for filepath in files:
            img = cv2.imread(filepath)
            if img is None:
                print(f"    WARNING: could not read {filepath}, skipping")
                continue

            # convert to grayscale right away
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            labels.append(label)

    print(f"  Total images loaded: {len(images)}")
    return images, np.array(labels)


# --- preprocess a single grayscale image for sklearn models ---

def preprocess_image(img, size=64):
    # resize to size x size
    resized = cv2.resize(img, (size, size))

    # flatten to 1D array
    flat = resized.flatten()

    # normalize pixel values to [0, 1]
    normalized = flat / 255.0

    return normalized


# --- get the best available torch device ---

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# --- save a confusion matrix plot ---

def save_confusion_matrix(y_true, y_pred, model_name, aug_type, output_dir):
    class_names = ["Word", "Google Docs", "Python"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - {aug_type}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # filename format: modelname_augtype.png
    filename = f"{model_name.lower()}_{aug_type.lower()}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


# --- print accuracy, precision, recall, f1, and classification report ---

def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n  {model_name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    class_names = ["Word (0)", "Google Docs (1)", "Python (2)"]
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # return accuracy so callers can use it
    return acc
