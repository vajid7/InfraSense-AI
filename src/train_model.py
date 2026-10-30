
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    fbeta_score
)

# =====================================
# PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"

RANDOM_STATE = 42

# =====================================
# LOAD DATA
# =====================================

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

target = "Machine failure"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found.")

excluded = [
    target, "TWF", "HDF", "PWF", "OSF", "RNF",
    "UDI", "Product ID"
]

X = df.drop(columns=excluded, errors="ignore")
y = df[target].astype(int)

if y.nunique() != 2:
    raise ValueError("Expected a binary failure target.")

print("Dataset shape:", df.shape)
print("Class distribution:")
print(y.value_counts().sort_index())

# =====================================
# TRAIN / VALIDATION / TEST SPLIT
# 60% / 20% / 20%
# =====================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# =====================================
# PREPROCESSING
# =====================================

categorical = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    ("num", "passthrough", numerical)
])

# =====================================
# MODEL EXPERIMENTS
# =====================================

experiments = {
    "baseline": None,
    "balanced": "balanced",
    "balanced_subsample": "balanced_subsample"
}

results = []
trained_models = {}

print("\nComparing Random Forest configurations...")

for name, class_weight in experiments.items():

    print(f"\nTraining experiment: {name}")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Threshold selection uses validation data only
    val_prob = pipeline.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )

    if len(thresholds) > 0:
        p = precision[:-1]
        r = recall[:-1]

        f2_scores = (
            5 * p * r
        ) / (
            4 * p + r + 1e-12
        )

        best_idx = int(f2_scores.argmax())
        threshold = float(thresholds[best_idx])
    else:
        threshold = 0.5

    val_pred = (val_prob >= threshold).astype(int)

    val_f2 = fbeta_score(
        y_val, val_pred,
        beta=2,
        zero_division=0
    )

    val_recall = (
        ((y_val == 1) & (val_pred == 1)).sum()
        / max((y_val == 1).sum(), 1)
    )

    val_precision = (
        ((y_val == 1) & (val_pred == 1)).sum()
        / max((val_pred == 1).sum(), 1)
    )

    results.append({
        "Experiment": name,
        "Threshold": threshold,
        "Validation F2": val_f2,
        "Validation Precision": val_precision,
        "Validation Recall": val_recall
    })

    trained_models[name] = {
        "model": pipeline,
        "threshold": threshold
    }

# =====================================
# SELECT MODEL USING VALIDATION F2
# =====================================

results_df = pd.DataFrame(results)

print("\nValidation comparison:")
print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.4f}".format,
            "Validation F2": "{:.4f}".format,
            "Validation Precision": "{:.4f}".format,
            "Validation Recall": "{:.4f}".format
        }
    )
)

best_row = results_df.loc[
    results_df["Validation F2"].idxmax()
]

best_name = best_row["Experiment"]
best_model = trained_models[best_name]["model"]
best_threshold = trained_models[best_name]["threshold"]

print("\nSelected model:", best_name)
print("Selected threshold:", round(best_threshold, 4))

# =====================================
# FINAL TEST EVALUATION
# Test set is not used for model selection
# =====================================

test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    y_test, test_pred,
    labels=[0, 1]
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, test_prob), 4
))

print("PR-AUC (Average Precision):", round(
    average_precision_score(y_test, test_prob), 4
))

print("Test F2:", round(
    fbeta_score(
        y_test, test_pred,
        beta=2,
        zero_division=0
    ), 4
))

# =====================================
# SAVE SELECTED MODEL
# Keep compatibility with dashboard/predict.py
# =====================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

artifact = {
    "model": best_model,
    "threshold": best_threshold,
    "features": X.columns.tolist(),
    "selected_experiment": best_name,
    "validation_results": results
}

joblib.dump(artifact, MODEL_PATH)

print("\nSelected model saved to:", MODEL_PATH)
print("Training and evaluation complete.")


