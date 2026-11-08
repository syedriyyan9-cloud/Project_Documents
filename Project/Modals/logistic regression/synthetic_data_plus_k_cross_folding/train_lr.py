import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------- CONFIG ----------
DATASET_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Training-7000.csv'

TARGET_COL = 'Result'
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

MODEL_NAME = 'LogisticRegression'
USE_SMOTE = False
# ----------------------------


# ---------- LOAD DATA ----------
df = pd.read_csv(DATASET_PATH)

df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)


# ---------- REMOVE TRACKING COLUMN ----------
if 'is_synthetic' in df.columns:
    df.drop(columns=['is_synthetic'], inplace=True)


# ---------- REMOVE DUPLICATES ----------
duplicates = df.duplicated().sum()

df.drop_duplicates(inplace=True)

print("Duplicates removed:", duplicates)


# ---------- REMOVE MISSING VALUES ----------
missing_before = df.isna().sum().sum()

df.dropna(inplace=True)

print("Missing values removed:", missing_before)
print("Dataset shape after cleaning:", df.shape)


# ---------- ENCODE GENDER ----------
print("Gender values:", df['Gender'].unique())

df['Gender'] = df['Gender'].astype(str).str.strip().map({
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
    raise ValueError("Target mapping failed.")


print(
    "Result after encoding:",
    df[TARGET_COL].value_counts().to_dict()
)


# ---------- SPLIT FEATURES AND TARGET ----------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# ---------- TRAIN-TEST SPLIT (80/20) ----------
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


# ---------- LOGISTIC REGRESSION MODEL ----------
# Scaling and Logistic Regression are kept inside one pipeline.
# This ensures the scaler is fitted only on each training fold.

lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ---------- K-FOLD CROSS VALIDATION ----------
# StratifiedKFold maintains class proportions in each fold.
# If SMOTE is enabled, it is applied only to each fold's
# training portion, never its validation portion.

if USE_SMOTE:
    cv_model = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
else:
    cv_model = lr


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'auc': 'roc_auc'
}

cv_results = cross_validate(
    cv_model,
    X_train,
    y_train,
    cv=skf,
    scoring=scoring,
    return_train_score=False
)


# ---------- CROSS VALIDATION RESULTS ----------
cv_metrics = {
    'CV_Accuracy_Mean': cv_results['test_accuracy'].mean(),
    'CV_Accuracy_Std': cv_results['test_accuracy'].std(),

    'CV_Precision_Mean': cv_results['test_precision'].mean(),
    'CV_Precision_Std': cv_results['test_precision'].std(),

    'CV_Recall_Mean': cv_results['test_recall'].mean(),
    'CV_Recall_Std': cv_results['test_recall'].std(),

    'CV_F1_Mean': cv_results['test_f1'].mean(),
    'CV_F1_Std': cv_results['test_f1'].std(),

    'CV_AUC_Mean': cv_results['test_auc'].mean(),
    'CV_AUC_Std': cv_results['test_auc'].std()
}

print("\n========== 5-FOLD CROSS VALIDATION ==========")

for metric, value in cv_metrics.items():
    print(f"{metric}: {value:.4f}")


# ---------- TRAIN FINAL LOGISTIC REGRESSION MODEL ----------
# Train on the complete 80% training split.

lr.fit(X_train, y_train)


# ---------- PREDICTIONS ----------
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]


# ---------- EVALUATION METRICS ----------
metrics = {
    'Model': MODEL_NAME,

    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC': roc_auc_score(y_test, y_prob),

    'TrainSize': len(X_train),
    'TestSize': len(X_test),

    'CV_Folds': N_SPLITS,
    'SMOTE_Enabled': USE_SMOTE,

    'RandomState': RANDOM_STATE
}

metrics.update(cv_metrics)


# ---------- CLASSIFICATION REPORT ----------
print(
    "\nClassification report:\n",
    classification_report(
        y_test,
        y_pred,
        target_names=['Negative', 'Positive'],
        zero_division=0
    )
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
        target_names=['Negative', 'Positive'],
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
joblib.dump(lr, f'{MODEL_NAME}_model.joblib')


print(
    f"\nDone. Files: "
    f"{MODEL_NAME}_metrics.xlsx, "
    f"{MODEL_NAME}_model.joblib"
)