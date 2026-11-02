import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE

# ---------- CONFIG ----------
DATA_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Dataset.csv'
TARGET_COL = 'Result'
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_NAME = 'LogisticRegression'
# ----------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Gender
print("Gender values:", df['Gender'].unique())
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
if df['Gender'].isna().any():
    raise ValueError("Gender mapping failed.")

# Target
print("Result values before:", df['Result'].unique())
df['Result'] = df['Result'].astype(str).str.strip().str.lower().map(
    {'positive': 1, 'negative': 0}
)
if df['Result'].isna().any():
    raise ValueError(f"Unmapped labels — found: {df['Result'].unique()}")
print("Result after encoding:", df['Result'].value_counts().to_dict())

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train original: {y_train.value_counts().to_dict()}")
print(f"Test original : {y_test.value_counts().to_dict()}")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Train after SMOTE: {y_train_res.value_counts().to_dict()}")

# LR needs feature scaling — wrap scaler + model in a Pipeline
# so the scaler is fitted on training data only (no leakage)
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])
lr.fit(X_train_res, y_train_res)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

metrics = {
    'Model': MODEL_NAME,
    'Accuracy':  accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall':    recall_score(y_test, y_pred),
    'F1':        f1_score(y_test, y_pred),
    'AUC':       roc_auc_score(y_test, y_prob),
    'TrainSize_Original':   len(y_train),
    'TrainSize_AfterSMOTE': len(y_train_res),
    'TestSize':             len(y_test),
    'RandomState':          RANDOM_STATE,
    'TestSizeRatio':        TEST_SIZE,
}

print("\nClassification report:\n", classification_report(y_test, y_pred))

metrics_df = pd.DataFrame([metrics])
cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['Actual_0', 'Actual_1'],
    columns=['Pred_0', 'Pred_1']
)
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).transpose()

with pd.ExcelWriter(f'{MODEL_NAME}_metrics.xlsx', engine='openpyxl') as writer:
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    cm_df.to_excel(writer, sheet_name='ConfusionMatrix')
    report_df.to_excel(writer, sheet_name='ClassificationReport')

joblib.dump(lr, f'{MODEL_NAME}_model.joblib')
print(f"\nDone. Files: {MODEL_NAME}_metrics.xlsx, {MODEL_NAME}_model.joblib")