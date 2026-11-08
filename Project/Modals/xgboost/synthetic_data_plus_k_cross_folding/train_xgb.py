import pandas as pd
import joblib
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------- CONFIG ----------
DATASET_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Training-7000.csv'

TARGET_COL = 'Result'
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
MODEL_NAME = 'XGBoost'

USE_SMOTE = False
# ----------------------------


# ---------- LOAD DATA ----------
df = pd.read_csv(DATASET_PATH)

df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)


# ---------- REMOVE TRACKING COLUMN ----------
# is_synthetic identifies generated records.
# It must not be included as a model feature.

if 'is_synthetic' in df.columns:
    df.drop(columns=['is_synthetic'], inplace=True)


# ---------- REMOVE DUPLICATES ----------
before_duplicates = len(df)

df.drop_duplicates(inplace=True)

print("Duplicate rows removed:", before_duplicates - len(df))


# ---------- REMOVE MISSING VALUES ----------
before_missing = len(df)

df.dropna(inplace=True)

print("Rows with missing values removed:", before_missing - len(df))

print("Clean dataset shape:", df.shape)


# ---------- ENCODE GENDER ----------
print("Gender values:", df['Gender'].unique())

df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})

if df['Gender'].isna().any():
    raise ValueError("Gender mapping failed.")


# ---------- ENCODE TARGET ----------
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
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("\nOriginal class distribution:", y.value_counts().to_dict())


# ---------- 80/20 TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining class distribution:", y_train.value_counts().to_dict())
print("Testing class distribution:", y_test.value_counts().to_dict())

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ---------- XGBOOST MODEL ----------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ---------- OPTIONAL SMOTE ----------
# SMOTE is disabled by default because the dataset
# may already contain synthetic records.
#
# If enabled, SMOTE is applied only within the
# training folds during cross-validation and to
# the final training split.
#
# The test data is NEVER SMOTEd.

if USE_SMOTE:
    model = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('xgb', xgb)
    ])
else:
    model = xgb


# ---------- 5-FOLD CROSS-VALIDATION ----------
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc'
}

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\n---------- CROSS-VALIDATION RESULTS ----------")

cv_metrics = {}

for metric in scoring.keys():

    scores = cv_results[f'test_{metric}']

    cv_metrics[f'CV_{metric.capitalize()}_Mean'] = scores.mean()
    cv_metrics[f'CV_{metric.capitalize()}_Std'] = scores.std()

    print(
        f"{metric.capitalize()}: "
        f"{scores.mean():.4f} (+/- {scores.std():.4f})"
    )


# ---------- TRAIN FINAL MODEL ----------
model.fit(X_train, y_train)


# ---------- PREDICTIONS ----------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ---------- EVALUATION METRICS ----------
metrics = {
    'Model': MODEL_NAME,

    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC': roc_auc_score(y_test, y_prob),

    'TrainSize_Original': len(y_train),
    'TrainSize_AfterSMOTE': (
        len(model.named_steps['smote'].fit_resample(X_train, y_train)[1])
        if USE_SMOTE else len(y_train)
    ),
    'TestSize': len(y_test),

    'RandomState': RANDOM_STATE
}

metrics.update(cv_metrics)


# ---------- CLASSIFICATION REPORT ----------
print(
    "\nClassification report:\n",
    classification_report(y_test, y_pred, zero_division=0)
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
        output_dict=True,
        zero_division=0
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
joblib.dump(model, f'{MODEL_NAME}_model.joblib')


print(
    f"\nDone. Files: "
    f"{MODEL_NAME}_metrics.xlsx, "
    f"{MODEL_NAME}_model.joblib"
)