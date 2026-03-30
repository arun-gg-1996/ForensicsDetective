# ForensicsDetective - Setup Guide

## What This Project Does

PDFs made by different apps (Word, Google Docs, Python) have different binary signatures.
I convert PDF binary data into grayscale PNG images and use machine learning to figure out
which app made each PDF.

**3 Classes:**
- Label 0 → Word (Microsoft Word)
- Label 1 → Google Docs
- Label 2 → Python (ReportLab)

**Dataset:**
- 894 original images total across 3 classes
- I apply 5 types of augmentations to each image (DPI makes 2 variants)
- That gives us 894 originals + 5364 augmented = 6258 total images

**Four Classifiers:**
- XGBoost — flattened 64x64 grayscale pixels as features
- SVM — same 64x64 flattened features, RBF kernel
- SGD — same 64x64 flattened features, modified huber loss
- CNN (PyTorch) — 128x128 grayscale images, 3 conv blocks

I train all four on original images only. The augmented images are just used to
test how well the classifiers hold up against distorted inputs.

---

## How to Set Up

You need Python 3.10 or higher.

Install all the libraries:

```bash
pip install -r requirements.txt
```

If you're on an Apple Silicon Mac, PyTorch will automatically use MPS
(Metal Performance Shaders) for faster training. Otherwise it uses CUDA or CPU.

---

## How to Run

Run everything from the `ForensicsDetective/` folder.

### Step 1 — Generate augmented images

```bash
python src/augmentation.py
```

This takes the 894 original images and creates 6 augmented versions of each one.
Everything gets saved to `data/augmented_images/`.

### Step 2 — Train all 4 classifiers

```bash
python src/classification.py
```

This trains XGBoost, SVM, SGD, and CNN one after the other on the original images
(80% train, 20% test, same split for all models). Saves all models and confusion
matrices to `results/`. Prints a summary table at the end.

You can also train classifiers individually if you prefer:
```bash
python src/xgboost_classifier.py   # just XGBoost
python src/svm_sgd_classifier.py   # just SVM and SGD
python src/cnn_classifier.py       # just CNN
```

### Step 3 — Run robustness analysis

```bash
python src/analysis.py
```

Tests all 4 trained models against all augmentation types.
Saves metrics CSV (with per-class and overall scores), confusion matrices,
and a robustness curve plot.

### Step 4 — Run statistical significance tests

```bash
python src/statistical_tests.py
```

Runs McNemar's test on all 6 classifier pairs (XGBoost vs SVM, XGBoost vs SGD, etc.)
to check if the accuracy differences are statistically significant.
Tests on both original and augmented data.
Saves results to `results/mcnemar_results.csv`.

---

## Results

### Baseline accuracy (tested on original images)

| Model   | Accuracy | Precision | Recall | F1     |
|---------|----------|-----------|--------|--------|
| XGBoost | 99.44%   | 99.47%    | 99.44% | 99.45% |
| SVM     | 97.77%   | 97.79%    | 97.77% | 97.76% |
| SGD     | 95.53%   | 95.94%    | 95.53% | 95.52% |
| CNN     | 98.32%   | 98.39%    | 98.32% | 98.32% |

### Robustness (accuracy on augmented images)

| Augmentation   | XGBoost | SVM    | SGD    | CNN    |
|----------------|---------|--------|--------|--------|
| Original       | 99.89%  | 99.55% | 99.11% | 99.66% |
| Gaussian Noise | 98.21%  | 99.55% | 98.99% | 99.66% |
| JPEG Compress  | 98.21%  | 99.55% | 99.22% | 99.66% |
| DPI Downsample | 86.13%  | 96.64% | 93.18% | 35.51% |
| Random Crop    | 48.10%  | 73.15% | 81.88% | 99.33% |
| Bit Depth      | 98.99%  | 99.66% | 99.11% | 26.29% |

### What I found
- SVM is the most robust model overall (highest average accuracy across augmentations)
- XGBoost and SVM both struggle with cropping since they use flattened pixels
- CNN struggles with bit depth and DPI changes since it loses texture detail
- CNN handles noise, JPEG, and cropping well since convolutions are good with spatial stuff
- All sklearn models (XGBoost, SVM, SGD) handle noise, JPEG, and bit depth well
- McNemar's test tells us which differences between classifiers are statistically significant

---

## File Structure

```
ForensicsDetective/
├── README.md                  # project overview and how to run
├── SETUP.md                   # this file
├── requirements.txt
├── .gitignore
│
├── word_pdfs_png/             # 398 Word PDF images
├── google_docs_pdfs_png/      # 396 Google Docs PDF images
├── python_pdfs_png/           # 100 Python PDF images
│
├── src/
│   ├── utils.py               # shared utility functions (load images, metrics, etc.)
│   ├── classification.py      # trains all 4 classifiers in one script
│   ├── augmentation.py        # creates augmented images
│   ├── xgboost_classifier.py  # trains XGBoost (standalone)
│   ├── svm_sgd_classifier.py  # trains SVM and SGD (standalone)
│   ├── cnn_classifier.py      # trains CNN (standalone)
│   ├── analysis.py            # robustness testing (all 4 models vs augmentations)
│   └── statistical_tests.py   # McNemar's significance tests (all 6 pairs)
│
├── data/
│   └── augmented_images/
│       ├── original/
│       ├── gaussian/
│       ├── jpeg/
│       ├── dpi/
│       ├── crop/
│       └── bitdepth/
│
├── results/
│   ├── xgboost_model.joblib
│   ├── svm_model.joblib
│   ├── sgd_model.joblib
│   ├── cnn_model.pth
│   ├── xgboost_test_results.csv
│   ├── cnn_test_results.csv
│   ├── performance_metrics.csv
│   ├── mcnemar_results.csv
│   ├── confusion_matrices/    # confusion matrix plots
│   └── robustness_plots/      # robustness curve (all 4 models)
│
└── reports/
    └── final_research_report.pdf
```

### What each src/ file does

| File | Purpose |
|------|---------|
| `utils.py` | Shared functions: image loading, preprocessing, device selection, confusion matrix saving, metrics printing. Other scripts import from here. |
| `classification.py` | Trains all 4 classifiers (XGBoost, SVM, SGD, CNN) in sequence on original images. Uses the same train/test split for all models. Saves all models and confusion matrices. |
| `augmentation.py` | Takes the 894 original PNGs and applies 5 augmentation types (gaussian noise, JPEG compression, DPI downsampling, random crop, bit depth reduction). Saves everything to `data/augmented_images/`. |
| `xgboost_classifier.py` | Standalone XGBoost training script. |
| `svm_sgd_classifier.py` | Standalone SVM + SGD training script. |
| `cnn_classifier.py` | Standalone CNN training script. Defines the `SimpleCNN` class that other scripts import. |
| `analysis.py` | Loads all 4 trained models and tests them on each augmentation type. Saves a CSV with all metrics and a robustness curve plot. |
| `statistical_tests.py` | Runs McNemar's test on all 6 classifier pairs to check if accuracy differences are statistically significant. Tests on both original and augmented data. Saves results to CSV. |
