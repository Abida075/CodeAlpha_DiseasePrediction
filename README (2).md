# Disease Prediction from Medical Data

Task 4 — CodeAlpha Machine Learning Internship

## What it does
Predicts whether a breast tumor is malignant or benign using diagnostic
measurements from the Breast Cancer Wisconsin dataset (built into
scikit-learn — no external download needed). Trains and compares two
classifiers, then reports the standard classification metrics.

## Files
- `disease_prediction.py` — loads data, trains models, evaluates, saves results
- `prediction_results.csv` — test set with actual vs predicted labels
- `model_comparison.csv` — side-by-side metrics for both models

## How to run
```bash
pip install pandas numpy scikit-learn
python disease_prediction.py
```

## Approach
- **Data**: 569 patient records, 30 numeric diagnostic features (cell size, texture, concavity, etc.), no missing values
- **Preprocessing**: 80/20 train-test split (stratified), feature scaling for Logistic Regression
- **Models**: Logistic Regression and Random Forest
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix

## Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.986 | 0.986 | 0.986 | 0.995 |
| Random Forest | 0.956 | 0.959 | 0.972 | 0.966 | 0.993 |

Logistic Regression slightly outperformed Random Forest on this dataset.
The most predictive features were tumor size and shape measurements
(worst perimeter, worst area, worst concave points).

## What I practiced
- Loading and exploring a real-world medical dataset
- Train/test splitting and feature scaling
- Training and comparing multiple classification algorithms
- Evaluating models with precision, recall, F1, and ROC-AUC (not just accuracy)
- Interpreting feature importance
