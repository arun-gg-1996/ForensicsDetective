# Assignment 2 — Student Submission

**Student:** Arun G Ghontale
**Course:** EAS 510 - Basics of AI
**Due:** March 28, 2026

## How to Run My Code

Install dependencies:
```bash
pip install -r requirements.txt
```

Run in this order:
```bash
python src/augmentation.py       # generate augmented images
python src/xgboost_classifier.py # train XGBoost
python src/cnn_classifier.py     # train CNN (requires Apple MPS)
python src/svm_sgd_classifier.py # train SVM and SGD
python src/analysis.py           # run robustness analysis
```

## My Results

| Augmentation | XGBoost | SVM | SGD | CNN |
|-------------|---------|-----|-----|-----|
| Original | 99.89% | 99.55% | 99.11% | 99.66% |
| Gaussian | 98.21% | 99.55% | 98.99% | 99.66% |
| JPEG | 98.21% | 99.55% | 99.22% | 99.66% |
| DPI | 86.13% | 96.64% | 93.18% | 35.51% |
| Crop | 48.10% | 73.15% | 81.88% | 99.33% |
| Bit Depth | 98.99% | 99.66% | 99.11% | 26.29% |

## Deliverables
- `src/` — all classifier and analysis scripts
- `results/` — confusion matrices, robustness plots, metrics CSV
- `reports/final_research_report.pdf` — full research report
- `SETUP.md` — environment setup instructions

---