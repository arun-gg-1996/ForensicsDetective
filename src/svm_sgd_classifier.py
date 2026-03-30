import cv2
import numpy as np
import os
import glob
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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
    print("SVM & SGD Classifiers - Training on Original Images")
    print("=" * 50)

    # create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)

    # --- step 1: load all images ---
    print("\nStep 1: Loading images...")

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

    # class names for reports and plots
    class_names = ["Word (0)", "Google Docs (1)", "Python (2)"]

    # =============================================
    # TRAIN SVM
    # =============================================
    print("\n" + "=" * 50)
    print("Training SVM Classifier...")
    print("  kernel='rbf', C=1.0, gamma='scale'")
    print("=" * 50)

    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        decision_function_shape='ovr',
        random_state=42,
    )

    svm_model.fit(X_train, y_train)
    print("SVM training complete!")

    # evaluate svm on test set
    print("\nEvaluating SVM on test set...")
    svm_pred = svm_model.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred, average='weighted')
    svm_rec = recall_score(y_test, svm_pred, average='weighted')
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')

    print(f"\n  Accuracy:  {svm_acc:.4f}")
    print(f"  Precision: {svm_prec:.4f}")
    print(f"  Recall:    {svm_rec:.4f}")
    print(f"  F1 Score:  {svm_f1:.4f}")

    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_pred, target_names=class_names))

    # save svm confusion matrix
    cm = confusion_matrix(y_test, svm_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices/svm_confusion.png", dpi=150)
    plt.close()
    print("  Saved confusion matrix to results/confusion_matrices/svm_confusion.png")

    # save svm model
    joblib.dump(svm_model, "results/svm_model.joblib")
    print("  Saved model to results/svm_model.joblib")

    # =============================================
    # TRAIN SGD
    # =============================================
    print("\n" + "=" * 50)
    print("Training SGD Classifier...")
    print("  max_iter=1000, loss='modified_huber'")
    print("=" * 50)

    sgd_model = SGDClassifier(
        max_iter=1000,
        random_state=42,
        tol=1e-3,
        loss='modified_huber',
    )

    sgd_model.fit(X_train, y_train)
    print("SGD training complete!")

    # evaluate sgd on test set
    print("\nEvaluating SGD on test set...")
    sgd_pred = sgd_model.predict(X_test)

    sgd_acc = accuracy_score(y_test, sgd_pred)
    sgd_prec = precision_score(y_test, sgd_pred, average='weighted')
    sgd_rec = recall_score(y_test, sgd_pred, average='weighted')
    sgd_f1 = f1_score(y_test, sgd_pred, average='weighted')

    print(f"\n  Accuracy:  {sgd_acc:.4f}")
    print(f"  Precision: {sgd_prec:.4f}")
    print(f"  Recall:    {sgd_rec:.4f}")
    print(f"  F1 Score:  {sgd_f1:.4f}")

    print("\nSGD Classification Report:")
    print(classification_report(y_test, sgd_pred, target_names=class_names))

    # save sgd confusion matrix
    cm = confusion_matrix(y_test, sgd_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("SGD Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrices/sgd_confusion.png", dpi=150)
    plt.close()
    print("  Saved confusion matrix to results/confusion_matrices/sgd_confusion.png")

    # save sgd model
    joblib.dump(sgd_model, "results/sgd_model.joblib")
    print("  Saved model to results/sgd_model.joblib")

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  SVM  - Accuracy: {svm_acc:.4f}  F1: {svm_f1:.4f}")
    print(f"  SGD  - Accuracy: {sgd_acc:.4f}  F1: {sgd_f1:.4f}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
