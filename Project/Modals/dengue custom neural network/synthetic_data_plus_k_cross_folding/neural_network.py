import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ---------- CONFIG ----------
DATASET_PATH = Path(__file__).resolve().parents[2] / 'Dengue-Training-7000.csv'

TARGET_COL = 'Result'
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

EPOCHS = 100
BATCH_SIZE = 32

MODEL_NAME = 'Custom_Shallow_NN'
# ----------------------------


# ---------- SET RANDOM SEED ----------
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


# ---------- LOAD DATA ----------
df = pd.read_csv(DATASET_PATH)

df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)


# ---------- REMOVE TRACKING COLUMN ----------
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

df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map({
    'male': 1,
    'female': 0
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


# ---------- SPLIT FEATURES AND TARGET ----------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# Ensure all input features are numerical.
X = X.apply(pd.to_numeric, errors='raise')

FEATURE_COLUMNS = X.columns.tolist()

print("\nClass distribution:", y.value_counts().to_dict())


# ---------- 80/20 TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("Training distribution:", y_train.value_counts().to_dict())
print("Testing distribution:", y_test.value_counts().to_dict())


# ---------- CUSTOM SHALLOW NEURAL NETWORK ----------
def build_model(input_features):

    model = Sequential([
        Input(shape=(input_features,)),

        Dense(32, activation='relu'),

        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ---------- EARLY STOPPING ----------
def get_early_stopping():

    return EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )


# ---------- 5-FOLD CROSS-VALIDATION ----------
print("\n---------- 5-FOLD CROSS-VALIDATION ----------")

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'auc': []
}


for fold, (train_index, val_index) in enumerate(
    cv.split(X_train, y_train), start=1
):

    print(f"\nTraining Fold {fold}/{N_SPLITS}")

    X_fold_train = X_train.iloc[train_index]
    X_fold_val = X_train.iloc[val_index]

    y_fold_train = y_train.iloc[train_index]
    y_fold_val = y_train.iloc[val_index]


    # Split fold training data for internal validation.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_fold_train,
        y_fold_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_fold_train
    )


    # Scale using only the fitting portion.
    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_fold_val_scaled = scaler.transform(X_fold_val)


    # Build a fresh model for every fold.
    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(RANDOM_STATE + fold)

    model = build_model(X_train.shape[1])


    # Train model.
    model.fit(
        X_fit_scaled,
        y_fit,
        validation_data=(X_val_scaled, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[get_early_stopping()],
        verbose=0
    )


    # Evaluate on the fold's held-out validation portion.
    y_fold_prob = model.predict(
        X_fold_val_scaled,
        verbose=0
    ).ravel()

    y_fold_pred = (y_fold_prob >= 0.5).astype(int)


    cv_results['accuracy'].append(
        accuracy_score(y_fold_val, y_fold_pred)
    )

    cv_results['precision'].append(
        precision_score(y_fold_val, y_fold_pred, zero_division=0)
    )

    cv_results['recall'].append(
        recall_score(y_fold_val, y_fold_pred, zero_division=0)
    )

    cv_results['f1'].append(
        f1_score(y_fold_val, y_fold_pred, zero_division=0)
    )

    cv_results['auc'].append(
        roc_auc_score(y_fold_val, y_fold_prob)
    )


    print(
        f"Fold {fold} Accuracy: "
        f"{cv_results['accuracy'][-1]:.4f}"
    )


# ---------- CROSS-VALIDATION SUMMARY ----------
print("\n---------- CROSS-VALIDATION RESULTS ----------")

cv_metrics = {}

for metric, scores in cv_results.items():

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    cv_metrics[f'CV_{metric.capitalize()}_Mean'] = mean_score
    cv_metrics[f'CV_{metric.capitalize()}_Std'] = std_score

    print(
        f"{metric.capitalize()}: "
        f"{mean_score:.4f} (+/- {std_score:.4f})"
    )


# ---------- FINAL MODEL TRAINING ----------
print("\n---------- FINAL MODEL TRAINING ----------")

# Keep a validation portion for early stopping.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)


# Fit scaler only on training portion.
final_scaler = StandardScaler()

X_fit_scaled = final_scaler.fit_transform(X_fit)
X_val_scaled = final_scaler.transform(X_val)
X_test_scaled = final_scaler.transform(X_test)


# Build final model.
tf.keras.backend.clear_session()

tf.keras.utils.set_random_seed(RANDOM_STATE)

final_model = build_model(X_train.shape[1])

final_model.summary()


# Train final model.
history = final_model.fit(
    X_fit_scaled,
    y_fit,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[get_early_stopping()],
    verbose=1
)


# ---------- TEST PREDICTIONS ----------
y_prob = final_model.predict(
    X_test_scaled,
    verbose=0
).ravel()

y_pred = (y_prob >= 0.5).astype(int)


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
    'RandomState': RANDOM_STATE
}

metrics.update(cv_metrics)


# ---------- CLASSIFICATION REPORT ----------
print(
    "\nClassification report:\n",
    classification_report(
        y_test,
        y_pred,
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


# ---------- SAVE NEURAL NETWORK ----------
final_model.save(f'{MODEL_NAME}_model.keras')


# ---------- SAVE SCALER ----------
joblib.dump(
    final_scaler,
    f'{MODEL_NAME}_scaler.joblib'
)


# ---------- SAVE FEATURE COLUMNS ----------
joblib.dump(
    FEATURE_COLUMNS,
    f'{MODEL_NAME}_features.joblib'
)


print("\nTraining and evaluation completed!")

print("Files saved:")
print(f"1. {MODEL_NAME}_model.keras")
print(f"2. {MODEL_NAME}_scaler.joblib")
print(f"3. {MODEL_NAME}_features.joblib")
print(f"4. {MODEL_NAME}_metrics.xlsx")