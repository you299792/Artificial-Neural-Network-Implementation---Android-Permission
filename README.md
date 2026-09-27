# Android Permission Classification — ANN Lab (Week 6-7)

Artificial Neural Network implementation for classifying Android apps as **Benign (0)** or **Malware (1)** based on their requested permissions, using the [Kaggle Android Permission Dataset](https://www.kaggle.com/datasets/saurabhshahane/android-permission-dataset).

## Overview

This project trains and compares four ANN architectures of decreasing depth to study the trade-off between model capacity, overfitting risk, and generalization on a real-world, imbalanced tabular dataset (~29,999 apps, ~157 numeric permission-flag features after cleaning, ~2:1 malware:benign ratio).

## Files

| File | Description |
|---|---|
| `Android_Permission.csv` | Raw Kaggle dataset (apps as rows, permissions as binary columns, plus metadata). |
| `ANN_Android_Permission_Solution.ipynb` | Primary implementation — procedural, step-by-step notebook following the lab workflow. |
| `ANN_Android_Permission_Alternate.ipynb` | Independent implementation — object-oriented data pipeline + Keras Functional API, config/loop-driven model comparison. |

## Requirements
python >= 3.9
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow >= 2.x


Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Workflow

1. **Load Data** — read the CSV, inspect shape, columns, dtypes, and raw class balance.
2. **Data Cleaning**
   - Drop non-numeric columns (app name, package, description, etc.).
   - Ensure the target (`Class`) is binary (0 = Benign, 1 = Malware).
   - Drop constant / zero-variance columns.
   - Handle missing values and duplicate rows.
3. **Train/Test Split** — stratified 80/20 split to preserve class ratio; feature scaling with `StandardScaler`; class weights computed to handle imbalance.
4. **Modeling** — four ANN architectures, each with ReLU + He-normal hidden layers, BatchNormalization, Dropout(0.2), and a sigmoid output:

   | Model | Architecture |
   |---|---|
   | Model 1 | 6 layers: 128, 64, 32, 16, 8, 1 |
   | Model 2 | 5 layers: 64, 32, 16, 8, 1 |
   | Model 3 | 4 layers: 32, 16, 8, 1 |
   | Model 4 | 3 layers: 16, 8, 1 |

   All models use `Adam` optimizer, `binary_crossentropy` loss, `EarlyStopping` (patience=8), and `ReduceLROnPlateau`.

5. **Evaluation & Comparison** — Accuracy, Precision, Recall, F1-Score, and Confusion Matrices for each model, plus training-history curves and a metrics heatmap.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ANN-D6 (128-64-32-16-8-1) | 63.02% | 84.19% | 48.57% | 61.60% |
| ANN-D5 (64-32-16-8-1)     | 63.07% | 83.25% | 49.47% | 62.06% |
| ANN-D4 (32-16-8-1)        | 64.00% | 81.49% | 53.11% | 64.31% |
| ANN-D3 (16-8-1)           | 66.24% | 70.09% | 77.99% | **73.83%** |

**Best model by F1-Score:** ANN-D3 (16-8-1) — 73.83%

## Discussion

Across all four architectures, accuracy stayed clustered in the 63–66% range, but F1-Score improved steadily as depth decreased. The two deepest models (D6, D5) show high precision but weak recall (~49%), meaning they are conservative and miss roughly half of actual malware apps — a sign that stacking more layers and regularization made optimization harder rather than adding useful capacity for this feature set. The simplest model (D3, 3 layers) performed best overall, with recall jumping to 77.99% at a moderate precision cost, indicating the benign/malware signal in this dataset is largely captured by a shallow non-linear boundary, and that additional depth was counterproductive rather than helpful here.

## Notes

- Class imbalance is handled via `class_weight='balanced'` during training rather than resampling, so every model trains on the full, unaltered training distribution.
- Numbers above will vary slightly with different random seeds, hardware, or TensorFlow versions.
- Run the modeling cells (Step 4–5) in an environment with TensorFlow installed (e.g., Google Colab or a local Python environment with `pip install tensorflow`).
