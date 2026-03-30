import numpy as np
import pandas as pd
import os
import sys
import cv2
import glob
import torch
import joblib
from itertools import combinations
from scipy.stats import chi2 as chi2_dist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# add src folder to path so I can import my modules
sys.path.append(os.path.dirname(__file__))
from utils import load_images_from_folders, preprocess_image, get_device
from cnn_classifier import SimpleCNN


# significance threshold
ALPHA = 0.05

# the 6 classifier pairs to compare
MODEL_NAMES = ["XGBoost", "SVM", "SGD", "CNN"]
MODEL_PAIRS = list(combinations(MODEL_NAMES, 2))


# --- McNemar's test ---
# compares two classifiers on the same test set by looking at
# where they DISAGREE (one right, one wrong)

def mcnemars_test(y_true, y_pred_a, y_pred_b):
    correct_a = (y_true == y_pred_a).astype(int)
    correct_b = (y_true == y_pred_b).astype(int)

    # 2x2 contingency table
    n11 = np.sum((correct_a == 1) & (correct_b == 1))  # both correct
    n10 = np.sum((correct_a == 1) & (correct_b == 0))  # A correct, B wrong
    n01 = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B correct
    n00 = np.sum((correct_a == 0) & (correct_b == 0))  # both wrong

    table = np.array([[n11, n10], [n01, n00]])

    # discordant pairs are what McNemar's test cares about
    b = n10  # A right, B wrong
    c = n01  # A wrong, B right

    if b + c == 0:
        # no disagreements at all
        return 0.0, 1.0, table

    # chi-squared with continuity correction (Edwards)
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = chi2_dist.sf(chi2_stat, df=1)

    return chi2_stat, p_value, table


# --- get predictions from a sklearn model on grayscale images ---

def predict_sklearn(model, gray_images):
    X = np.array([preprocess_image(img) for img in gray_images])
    return model.predict(X)


# --- get predictions from CNN on grayscale images ---

def predict_cnn(model, gray_images, device):
    model.eval()
    all_preds = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(gray_images), batch_size):
            batch_imgs = gray_images[i:i + batch_size]
            tensors = []
            for img in batch_imgs:
                resized = cv2.resize(img, (128, 128))
                tensor = torch.FloatTensor(resized / 255.0).unsqueeze(0)
                tensors.append(tensor)
            batch = torch.stack(tensors).to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds)


# --- load augmented images from one augmentation folder ---
# returns grayscale images (not BGR) so I can reuse predict functions

def load_augmented_grayscale(aug_folder):
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
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(label)

    return images, np.array(labels)


# --- run McNemar's for all 6 pairs and return rows ---

def run_all_pairs(y_true, predictions, data_label):
    rows = []

    for name_a, name_b in MODEL_PAIRS:
        pred_a = predictions[name_a]
        pred_b = predictions[name_b]

        chi2_stat, p_value, table = mcnemars_test(y_true, pred_a, pred_b)
        significant = "Yes" if p_value < ALPHA else "No"

        # figure out which model did better in the discordant pairs
        b = table[0, 1]  # A correct, B wrong
        c = table[1, 0]  # A wrong, B correct
        if b > c:
            better = name_a
        elif c > b:
            better = name_b
        else:
            better = "Tie"

        acc_a = accuracy_score(y_true, pred_a)
        acc_b = accuracy_score(y_true, pred_b)

        rows.append({
            "data_type": data_label,
            "model_a": name_a,
            "model_b": name_b,
            "acc_a": round(acc_a, 4),
            "acc_b": round(acc_b, 4),
            "both_correct": int(table[0, 0]),
            "a_right_b_wrong": int(table[0, 1]),
            "a_wrong_b_right": int(table[1, 0]),
            "both_wrong": int(table[1, 1]),
            "chi2": round(chi2_stat, 4),
            "p_value": round(p_value, 6),
            "significant": significant,
            "better_model": better,
        })

    return rows


# --- print McNemar's results as a table ---

def print_results_table(rows, title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header = (f"  {'Pair':<22} {'Acc A':>7} {'Acc B':>7} "
              f"{'b':>4} {'c':>4} {'chi2':>8} {'p-value':>10} {'Sig?':>5} {'Better':>10}")
    print(header)
    print(f"  {'-' * 76}")

    for r in rows:
        pair = f"{r['model_a']} vs {r['model_b']}"
        print(f"  {pair:<22} {r['acc_a']:>7.4f} {r['acc_b']:>7.4f} "
              f"{r['a_right_b_wrong']:>4} {r['a_wrong_b_right']:>4} "
              f"{r['chi2']:>8.4f} {r['p_value']:>10.6f} "
              f"{'Yes' if r['significant'] == 'Yes' else 'No':>5} "
              f"{r['better_model']:>10}")

    # legend
    print(f"\n  b = A correct & B wrong,  c = A wrong & B correct  (discordant pairs)")
    print(f"  Significant at alpha = {ALPHA}")


# --- main script ---

def main():
    print("=" * 80)
    print("McNemar's Statistical Significance Tests")
    print("Comparing all 4 classifiers pairwise")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)

    # =============================================
    # STEP 1: LOAD ORIGINAL IMAGES AND SPLIT
    # =============================================
    print("\nStep 1: Loading original images and recreating test split...")
    images, labels = load_images_from_folders(".")
    print(f"Class distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}")

    # same split used during training across all classifiers
    indices = np.arange(len(images))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )

    test_images = [images[i] for i in idx_test]
    y_true = labels[idx_test]
    print(f"Test set size: {len(test_images)} images")

    # =============================================
    # STEP 2: LOAD ALL 4 MODELS
    # =============================================
    print("\nStep 2: Loading trained models...")

    xgb_model = joblib.load("results/xgboost_model.joblib")
    print("  Loaded XGBoost")

    svm_model = joblib.load("results/svm_model.joblib")
    print("  Loaded SVM")

    sgd_model = joblib.load("results/sgd_model.joblib")
    print("  Loaded SGD")

    device = get_device()
    cnn_model = SimpleCNN().to(device)
    cnn_model.load_state_dict(torch.load("results/cnn_model.pth", map_location=device))
    cnn_model.eval()
    print(f"  Loaded CNN (device: {device})")

    # =============================================
    # STEP 3: GENERATE PREDICTIONS ON ORIGINAL TEST SET
    # =============================================
    print("\nStep 3: Generating predictions on original test set...")

    predictions = {}

    print("  Predicting with XGBoost...")
    predictions["XGBoost"] = predict_sklearn(xgb_model, test_images)

    print("  Predicting with SVM...")
    predictions["SVM"] = predict_sklearn(svm_model, test_images)

    print("  Predicting with SGD...")
    predictions["SGD"] = predict_sklearn(sgd_model, test_images)

    print("  Predicting with CNN...")
    predictions["CNN"] = predict_cnn(cnn_model, test_images, device)

    # quick accuracy check
    for name in MODEL_NAMES:
        acc = accuracy_score(y_true, predictions[name])
        print(f"    {name}: {acc:.4f}")

    # =============================================
    # STEP 4: MCNEMAR'S TEST ON ORIGINAL DATA
    # =============================================
    print("\nStep 4: Running McNemar's test on original test set...")

    all_results = []

    original_rows = run_all_pairs(y_true, predictions, "original")
    all_results.extend(original_rows)
    print_results_table(original_rows, "McNemar's Test Results — Original Images")

    # =============================================
    # STEP 5: MCNEMAR'S TEST ON AUGMENTED DATA
    # =============================================
    augmentation_types = ["gaussian", "jpeg", "dpi", "crop", "bitdepth"]
    aug_base = "data/augmented_images"

    if os.path.isdir(aug_base):
        print(f"\nStep 5: Running McNemar's test on augmented data...")
        print(f"  Augmentation base: {aug_base}")

        for aug_type in augmentation_types:
            aug_folder = os.path.join(aug_base, aug_type)

            if not os.path.isdir(aug_folder):
                print(f"\n  WARNING: {aug_folder} not found, skipping {aug_type}")
                continue

            print(f"\n  Loading {aug_type} images...")
            aug_images, aug_labels = load_augmented_grayscale(aug_folder)
            print(f"    Loaded {len(aug_images)} images")

            # get predictions from all 4 models
            aug_preds = {}
            aug_preds["XGBoost"] = predict_sklearn(xgb_model, aug_images)
            aug_preds["SVM"] = predict_sklearn(svm_model, aug_images)
            aug_preds["SGD"] = predict_sklearn(sgd_model, aug_images)
            aug_preds["CNN"] = predict_cnn(cnn_model, aug_images, device)

            # run mcnemar's for all pairs
            aug_rows = run_all_pairs(aug_labels, aug_preds, aug_type)
            all_results.extend(aug_rows)
            print_results_table(aug_rows, f"McNemar's Test Results — {aug_type.upper()} Augmentation")
    else:
        print(f"\nStep 5: Augmented data not found at {aug_base}")
        print("  Run src/augmentation.py first to generate augmented images.")

    # =============================================
    # STEP 6: SAVE RESULTS TO CSV
    # =============================================
    print(f"\n\nStep 6: Saving results to CSV...")

    results_df = pd.DataFrame(all_results)
    csv_path = "results/mcnemar_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    print(f"  Total comparisons: {len(results_df)}")

    # =============================================
    # STEP 7: OVERALL SUMMARY
    # =============================================
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    # count significant differences per pair across all data types
    sig_df = results_df[results_df["significant"] == "Yes"]
    not_sig_df = results_df[results_df["significant"] == "No"]

    print(f"\nTotal tests run: {len(results_df)}")
    print(f"Significant differences found: {len(sig_df)}")
    print(f"Not significant: {len(not_sig_df)}")

    # significant pairs on original data
    orig_sig = sig_df[sig_df["data_type"] == "original"]
    orig_not_sig = not_sig_df[not_sig_df["data_type"] == "original"]

    print(f"\n--- Original Data ---")
    if len(orig_sig) > 0:
        print("  Significantly different pairs:")
        for _, row in orig_sig.iterrows():
            print(f"    {row['model_a']} vs {row['model_b']}: "
                  f"p={row['p_value']:.6f}, better={row['better_model']}")
    else:
        print("  No significant differences found on original data.")

    if len(orig_not_sig) > 0:
        print("  NOT significantly different pairs:")
        for _, row in orig_not_sig.iterrows():
            print(f"    {row['model_a']} vs {row['model_b']}: p={row['p_value']:.6f}")

    # count wins per model across ALL tests
    print(f"\n--- Win Count (across all data types) ---")
    print("  (How often each model was 'better' in significant tests)")

    win_counts = {name: 0 for name in MODEL_NAMES}
    for _, row in sig_df.iterrows():
        if row["better_model"] in win_counts:
            win_counts[row["better_model"]] += 1

    for name in MODEL_NAMES:
        print(f"    {name}: {win_counts[name]} significant wins")

    if max(win_counts.values()) > 0:
        best = max(win_counts, key=win_counts.get)
        print(f"\n  Model with most significant wins: {best} ({win_counts[best]} wins)")
    else:
        print(f"\n  No significant differences detected between any classifiers.")

    # per-augmentation summary
    if os.path.isdir(aug_base):
        print(f"\n--- Robustness Summary ---")
        for aug_type in augmentation_types:
            aug_sig = sig_df[sig_df["data_type"] == aug_type]
            if len(aug_sig) > 0:
                print(f"\n  {aug_type.upper()}: {len(aug_sig)} significant pair(s)")
                for _, row in aug_sig.iterrows():
                    print(f"    {row['model_a']} vs {row['model_b']}: "
                          f"p={row['p_value']:.6f}, better={row['better_model']}")
            else:
                print(f"\n  {aug_type.upper()}: No significant differences")

    print(f"\nNote: McNemar's test uses chi-squared with continuity correction.")
    print(f"Significance level: alpha = {ALPHA}")
    print(f"Results saved to: {csv_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
