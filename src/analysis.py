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

# add src folder to path so I can import the CNN class
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

def preprocess_for_sklearn(img):
    # same preprocessing for xgboost, svm, and sgd
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


# --- evaluate sklearn model (works for xgboost, svm, sgd) ---

def evaluate_sklearn(model, images, labels):
    # preprocess all images
    X = np.array([preprocess_for_sklearn(img) for img in images])
    y = np.array(labels)

    # predict
    y_pred = model.predict(X)

    # compute overall metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # compute per-class metrics
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    return y, y_pred, acc, prec, rec, f1, report


# --- evaluate cnn model ---

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

    # compute overall metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # compute per-class metrics
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)

    return y, y_pred, acc, prec, rec, f1, report


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


# --- helper to add results rows (overall + per-class) ---

def collect_results(model_name, aug_type, acc, prec, rec, f1, report):
    class_label_map = {"0": "Word", "1": "Google Docs", "2": "Python"}
    rows = []

    # overall row
    rows.append({
        "model": model_name,
        "augmentation_type": aug_type,
        "class_label": "overall",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    })

    # per-class rows
    for key, name in class_label_map.items():
        if key in report:
            rows.append({
                "model": model_name,
                "augmentation_type": aug_type,
                "class_label": name,
                "accuracy": round(acc, 4),
                "precision": round(report[key]["precision"], 4),
                "recall": round(report[key]["recall"], 4),
                "f1": round(report[key]["f1-score"], 4),
            })

    return rows


# --- main script ---

def main():
    print("=" * 60)
    print("Robustness Analysis - All 4 Models vs Augmented Images")
    print("=" * 60)

    # create output directories
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/robustness_plots", exist_ok=True)

    # --- step 1: load all 4 trained models ---
    print("\nStep 1: Loading trained models...")

    # load xgboost
    xgb_model = joblib.load("results/xgboost_model.joblib")
    print("  Loaded XGBoost from results/xgboost_model.joblib")

    # load svm
    svm_model = joblib.load("results/svm_model.joblib")
    print("  Loaded SVM from results/svm_model.joblib")

    # load sgd
    sgd_model = joblib.load("results/sgd_model.joblib")
    print("  Loaded SGD from results/sgd_model.joblib")

    # load cnn
    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load("results/cnn_model.pth", map_location=device))
    cnn_model.eval()
    print("  Loaded CNN from results/cnn_model.pth")

    # --- step 2: define what to test ---
    augmentation_types = ["original", "gaussian", "jpeg", "dpi", "crop", "bitdepth"]
    base_path = "data/augmented_images"

    # sklearn models use the same preprocessing
    sklearn_models = [
        ("XGBoost", xgb_model),
        ("SVM", svm_model),
        ("SGD", sgd_model),
    ]

    # store all results
    all_results = []

    # track accuracies for robustness curve
    accuracies = {
        "XGBoost": [],
        "SVM": [],
        "SGD": [],
        "CNN": [],
    }

    # --- step 3: evaluate all models on each augmentation ---
    print("\nStep 3: Evaluating all models on each augmentation type...\n")

    for aug_type in augmentation_types:
        aug_folder = os.path.join(base_path, aug_type)
        print(f"{'='*50}")
        print(f"Testing on: {aug_type}")
        print(f"{'='*50}")

        # load images for this augmentation
        images, labels = load_augmented_images(aug_folder)
        print(f"  Loaded {len(images)} images")

        # evaluate each sklearn model (xgboost, svm, sgd)
        for model_name, model in sklearn_models:
            print(f"\n  Evaluating {model_name}...")
            y_true, y_pred, acc, prec, rec, f1, report = \
                evaluate_sklearn(model, images, labels)

            print(f"    Accuracy: {acc:.4f}  Precision: {prec:.4f}  "
                  f"Recall: {rec:.4f}  F1: {f1:.4f}")

            # per-class summary
            for cls_key, cls_name in [("0", "Word"), ("1", "Google Docs"), ("2", "Python")]:
                if cls_key in report:
                    r = report[cls_key]
                    print(f"    {cls_name:15s} P:{r['precision']:.4f}  "
                          f"R:{r['recall']:.4f}  F1:{r['f1-score']:.4f}")

            accuracies[model_name].append(acc)

            # save confusion matrix
            cm_path = f"results/confusion_matrices/{model_name.lower()}_{aug_type}.png"
            save_confusion_matrix(y_true, y_pred,
                                  f"{model_name} - {aug_type}", cm_path)

            # collect results (overall + per-class)
            all_results.extend(collect_results(model_name, aug_type, acc, prec, rec, f1, report))

        # evaluate cnn
        print(f"\n  Evaluating CNN...")
        y_true, y_pred, acc, prec, rec, f1, report = \
            evaluate_cnn(cnn_model, images, labels)

        print(f"    Accuracy: {acc:.4f}  Precision: {prec:.4f}  "
              f"Recall: {rec:.4f}  F1: {f1:.4f}")

        for cls_key, cls_name in [("0", "Word"), ("1", "Google Docs"), ("2", "Python")]:
            if cls_key in report:
                r = report[cls_key]
                print(f"    {cls_name:15s} P:{r['precision']:.4f}  "
                      f"R:{r['recall']:.4f}  F1:{r['f1-score']:.4f}")

        accuracies["CNN"].append(acc)

        # save cnn confusion matrix
        cm_path = f"results/confusion_matrices/cnn_{aug_type}.png"
        save_confusion_matrix(y_true, y_pred,
                              f"CNN - {aug_type}", cm_path)

        # collect cnn results
        all_results.extend(collect_results("CNN", aug_type, acc, prec, rec, f1, report))

        print()

    # --- step 4: save all metrics to csv ---
    print("\nStep 4: Saving performance metrics to CSV...")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/performance_metrics.csv", index=False)
    print("  Saved to results/performance_metrics.csv")
    print(f"  Total rows: {len(results_df)} ({len(results_df[results_df['class_label']=='overall'])} overall + per-class)")

    # print overall results table
    print("\nOverall accuracy summary:")
    overall_df = results_df[results_df["class_label"] == "overall"]
    pivot = overall_df.pivot(index="augmentation_type", columns="model", values="accuracy")
    pivot = pivot.reindex(augmentation_types)
    pivot = pivot[["XGBoost", "SVM", "SGD", "CNN"]]
    print(pivot.to_string())

    # --- step 5: generate robustness curve plot ---
    print("\n\nStep 5: Generating robustness curve plot...")

    plt.figure(figsize=(12, 7))
    plt.plot(augmentation_types, accuracies["XGBoost"], 'o-', label='XGBoost', linewidth=2, markersize=8)
    plt.plot(augmentation_types, accuracies["SVM"], 's-', label='SVM', linewidth=2, markersize=8)
    plt.plot(augmentation_types, accuracies["SGD"], '^-', label='SGD', linewidth=2, markersize=8)
    plt.plot(augmentation_types, accuracies["CNN"], 'D-', label='CNN', linewidth=2, markersize=8)
    plt.xlabel("Augmentation Type")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve: Accuracy vs Augmentation Type (All 4 Models)")
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

    model_names = ["XGBoost", "SVM", "SGD", "CNN"]

    # find biggest accuracy drop for each model
    for name in model_names:
        baseline = accuracies[name][0]
        drops = [(aug, baseline - acc) for aug, acc in
                 zip(augmentation_types[1:], accuracies[name][1:])]
        worst = max(drops, key=lambda x: x[1])
        print(f"\n{name}:")
        print(f"  Baseline accuracy (original): {baseline:.4f}")
        print(f"  Biggest drop: {worst[0]} (dropped by {worst[1]:.4f} to {baseline - worst[1]:.4f})")

    # find best performing model (highest average accuracy across all augmentations)
    avg_accs = {name: np.mean(accuracies[name]) for name in model_names}
    best_model = max(avg_accs, key=avg_accs.get)

    print(f"\nAverage accuracy across all augmentations:")
    for name in model_names:
        print(f"  {name:10s}: {avg_accs[name]:.4f}")

    print(f"\nBest overall model: {best_model} (avg accuracy: {avg_accs[best_model]:.4f})")

    # find most robust model (smallest drop from baseline to worst case)
    min_accs = {name: min(accuracies[name]) for name in model_names}
    drops_from_baseline = {name: accuracies[name][0] - min_accs[name] for name in model_names}
    most_robust = min(drops_from_baseline, key=drops_from_baseline.get)

    print(f"Most robust model: {most_robust} (smallest max drop: {drops_from_baseline[most_robust]:.4f})")

    # full comparison table
    print(f"\nFull accuracy comparison:")
    print(f"  {'Augmentation':15s}", end="")
    for name in model_names:
        print(f"  {name:>8s}", end="")
    print()
    for i, aug in enumerate(augmentation_types):
        print(f"  {aug:15s}", end="")
        for name in model_names:
            print(f"  {accuracies[name][i]:8.4f}", end="")
        print()

    print("\nAll done!")


if __name__ == "__main__":
    main()
