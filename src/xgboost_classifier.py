import cv2
import numpy as np
import os
import glob
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd


def load_images(folder, label):
    # load all png images from a folder and assign a label
    images = []
    labels = []
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    print(f"  Loading {len(files)} images from {folder} (label={label})")

    for filepath in files:
        # read the image
        img = cv2.imread(filepath)
        if img is None:
            print(f"    WARNING: could not read {filepath}, skipping")
            continue

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize to 64x64
        resized = cv2.resize(gray, (64, 64))

        # flatten to 1D array (4096 features)
        flat = resized.flatten()

        # normalize pixel values to [0, 1]
        normalized = flat / 255.0

        images.append(normalized)
        labels.append(label)

    return images, labels


def main():
    print("=" * 50)
    print("XGBoost Classifier - Training on Original Images")
    print("=" * 50)

    # create output directories if they dont exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # --- step 1: load all images ---
    print("\nStep 1: Loading images...")

    # source folders and their labels
    folders = [
        ("word_pdfs_png", 0),
        ("google_docs_pdfs_png", 1),
        ("python_pdfs_png", 2),
    ]

    all_images = []
    all_labels = []

    for folder, label in folders:
        images, labels = load_images(folder, label)
        all_images.extend(images)
        all_labels.extend(labels)

    # convert to numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}, 2={np.sum(y==2)}")

    # --- step 2: split into train and test ---
    print("\nStep 2: Splitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")

    # --- step 3: train xgboost classifier ---
    print("\nStep 3: Training XGBoost classifier...")
    print("  n_estimators=100, max_depth=6, learning_rate=0.1")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42,
    )

    # train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
    )
    print("Training complete!")

    # --- step 4: evaluate on test set ---
    print("\nStep 4: Evaluating on test set...")
    y_pred = model.predict(X_test)

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

    # --- step 5: save confusion matrix plot ---
    print("Step 5: Saving confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices/xgboost_confusion.png", dpi=150)
    plt.close()
    print("  Saved to results/confusion_matrices/xgboost_confusion.png")

    # --- step 6: save the trained model ---
    print("\nStep 6: Saving trained model...")
    joblib.dump(model, "results/xgboost_model.joblib")
    print("  Saved to results/xgboost_model.joblib")

    # --- step 7: save test results to csv ---
    print("\nStep 7: Saving test predictions to CSV...")
    results_df = pd.DataFrame({
        "true_label": y_test,
        "predicted_label": y_pred,
    })
    results_df.to_csv("results/xgboost_test_results.csv", index=False)
    print("  Saved to results/xgboost_test_results.csv")

    print("\nAll done!")


if __name__ == "__main__":
    main()
