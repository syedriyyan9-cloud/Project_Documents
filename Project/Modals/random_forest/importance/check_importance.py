import joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / 'RandomForest_model.joblib'
print("Looking for:", MODEL_PATH, "| exists:", MODEL_PATH.exists())

rf = joblib.load(MODEL_PATH)
df = pd.read_csv(Path(__file__).resolve().parents[2] / 'Dengue-Dataset.csv')
df.columns = df.columns.str.strip()
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
X = df.drop(columns=['Result'])

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance.to_string(index=False))
importance.to_excel('RandomForest_feature_importance.xlsx', index=False)