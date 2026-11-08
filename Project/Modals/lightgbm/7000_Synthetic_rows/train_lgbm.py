
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# ---------- CONFIG ----------
TRAIN_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Training-7000.csv'
TEST_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Test-305.csv'

TARGET_COL = 'Result'
RANDOM_STATE = 42
MODEL_NAME = 'LightGBM'
# ----------------------------


# ---------- LOAD DATA ----------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)


# ---------- REMOVE TRACKING COLUMN ----------
# is_synthetic is only used to identify generated records.
# It must not be included as a model feature.

for df in [train_df, test_df]:
    if 'is_synthetic' in df.columns:
        df.drop(columns=['is_synthetic'], inplace=True)


# ---------- ENCODE GENDER ----------
for df in [train_df, test_df]:

    print("Gender values:", df['Gender'].unique())

    df['Gender'] = df['Gender'].map({
        'Male': 1,
        'Female': 0
    })

    if df['Gender'].isna().any():
        raise ValueError("Gender mapping failed.")


# ---------- ENCODE TARGET ----------
for df in [train_df, test_df]:

    print("Result values before:", df[TARGET_COL].unique())

    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            'positive': 1,
            'negative': 0
        })
    )

    if df[TARGET_COL].isna().any():
        raise ValueError(
            f"Unmapped labels — found: {df[TARGET_COL].unique()}"
        )

    print(
        "Result after encoding:",
        df[TARGET_COL].value_counts().to_dict()
    )


# ---------- SPLIT FEATURES AND TARGET ----------
# Training data is used only for training.
# Testing data is kept completely separate.

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

print(f"\nTrain original: {y_train.value_counts().to_dict()}")
print(f"Test original : {y_test.value_counts().to_dict()}")

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ---------- SMOTE (OPTIONAL) ----------
# Your 7,000-row training dataset already contains
# synthetic records generated during dataset preparation.
#
# Therefore, SMOTE is DISABLED by default.
#
# If you want to balance the training classes to 50:50,
# uncomment the following three lines.
#
# IMPORTANT:
# 1. Apply SMOTE ONLY to training data.
# 2. NEVER apply SMOTE to testing data.
# 3. SMOTE is optional and may not improve test performance.

# smote = SMOTE(random_state=RANDOM_STATE)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# print(f"Train after SMOTE: {y_train_res.value_counts().to_dict()}")

# Without SMOTE, use the existing training data.
X_train_res, y_train_res = X_train, y_train

# --------------------------------------


# ---------- TRAIN LIGHTGBM ----------
lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    random_state=RANDOM_STATE,
    verbose=-1
)

lgbm.fit(X_train_res, y_train_res)


# ---------- PREDICTIONS ----------
y_pred = lgbm.predict(X_test)
y_prob = lgbm.predict_proba(X_test)[:, 1]


# ---------- EVALUATION METRICS ----------
metrics = {
    'Model': MODEL_NAME,

    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_prob),

    'TrainSize_Original': len(y_train),
    'TrainSize_AfterSMOTE': len(y_train_res),
    'TestSize': len(y_test),

    'RandomState': RANDOM_STATE
}


# ---------- CLASSIFICATION REPORT ----------
print(
    "\nClassification report:\n",
    classification_report(y_test, y_pred)
)


# ---------- CREATE REPORT DATAFRAMES ----------
metrics_df = pd.DataFrame([metrics])

cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=['Actual_0', 'Actual_1'],
    columns=['Pred_0', 'Pred_1']
)

report_df = pd.DataFrame(
    classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
).transpose()


# ---------- SAVE EXCEL REPORT ----------
with pd.ExcelWriter(
    f'{MODEL_NAME}_metrics.xlsx',
    engine='openpyxl'
) as writer:

    metrics_df.to_excel(
        writer,
        sheet_name='Metrics',
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name='ConfusionMatrix'
    )

    report_df.to_excel(
        writer,
        sheet_name='ClassificationReport'
    )


# ---------- SAVE MODEL ----------
joblib.dump(lgbm, f'{MODEL_NAME}_model.joblib')


print(
    f"\nDone. Files: "
    f"{MODEL_NAME}_metrics.xlsx, "
    f"{MODEL_NAME}_model.joblib"
)
