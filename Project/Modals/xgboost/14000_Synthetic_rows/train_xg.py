import pandas as pd
import joblib
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

# ==============================
# Configuration
# ==============================

DATASET_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Training-14000-Balanced.csv'
MODEL_NAME = "XGBoost"
TARGET_COL = "Result"
RANDOM_STATE = 42

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv(DATASET_PATH)

# Remove synthetic tracking column
df = df.drop(columns=["is_synthetic"], errors="ignore")

# Encode Gender
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

if df["Gender"].isna().any():
    raise ValueError("Unexpected or missing Gender values.")

# Encode Target
df[TARGET_COL] = (
    df[TARGET_COL]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "positive": 1,
        "negative": 0
    })
)

if df[TARGET_COL].isna().any():
    raise ValueError("Unexpected or missing Result values.")

# ==============================
# Separate Features and Target
# ==============================

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ==============================
# Train-Test Split (80/20)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# Train XGBoost Model
# ==============================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# Evaluation Metrics
# ==============================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL RESULTS ==========")
print("Model:", MODEL_NAME)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# Classification Report
# ==============================

report = classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

# ==============================
# Save Results to Excel
# ==============================

metrics_df = pd.DataFrame([{
    "Model": MODEL_NAME,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUC": auc,
    "Training Samples": len(X_train),
    "Testing Samples": len(X_test),
    "Random State": RANDOM_STATE
}])

confusion_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)

report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )
).transpose()

with pd.ExcelWriter("XGBoost_Results.xlsx") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    confusion_df.to_excel(writer, sheet_name="ConfusionMatrix")
    report_df.to_excel(writer, sheet_name="ClassificationReport")

# ==============================
# Save Trained Model
# ==============================

joblib.dump(model, "XGBoost_Model.joblib")

print("\nResults saved successfully.")
print("Model saved successfully.")