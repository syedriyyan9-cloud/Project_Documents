# import pandas as pd

# file_path_xgboost_metrices = "Modals/xgboost/synthetic_data_plus_k_cross_folding/XGBoost_metrics.xlsx"
# file_path_random_forest_metrices = "Modals/random_forest/synthetic_data_plus_k_cross_folding/RandomForest_metrics.xlsx"
# file_path_logistic_regression_metrices = "Modals/logistic regression/synthetic_data_plus_k_cross_folding/LogisticRegression_metrics.xlsx"
# file_path_lightgbm_metrices = "Modals/lightgbm/synthetic_data_plus_k_cross_folding/LightGBM_metrics.xlsx"

# xgboost_df = pd.read_excel(file_path_xgboost_metrices)
# random_forest_df = pd.read_excel(file_path_random_forest_metrices) 
# logistic_regression_df = pd.read_excel(file_path_logistic_regression_metrices)
# lightgbm_df = pd.read_excel(file_path_lightgbm_metrices)

# print("XGBoost")
# print(xgboost_df.to_string())
# print("------------------------")

# print("Random Forest")
# print(random_forest_df.to_string())
# print("------------------------")

# print("Logistic Regression")
# print(logistic_regression_df.to_string())
# print("------------------------")

# print("LightGBM")
# print(lightgbm_df.to_string())
# print("------------------------")



import pandas as pd
from pathlib import Path


# ---------- FILE PATHS ----------
file_path_xgboost_metrics = "Modals/xgboost/synthetic_data_plus_k_cross_folding/XGBoost_metrics.xlsx"

file_path_random_forest_metrics = "Modals/random_forest/synthetic_data_plus_k_cross_folding/RandomForest_metrics.xlsx"

file_path_logistic_regression_metrics = "Modals/logistic regression/synthetic_data_plus_k_cross_folding/LogisticRegression_metrics.xlsx"

file_path_lightgbm_metrics = "Modals/lightgbm/synthetic_data_plus_k_cross_folding/LightGBM_metrics.xlsx"

file_path_custom_neural_network = "Modals/dengue custom neural network/synthetic_data_plus_k_cross_folding/Custom_Shallow_NN_metrics.xlsx"

# ---------- LOAD EXCEL FILES ----------
xgboost_df = pd.read_excel(file_path_xgboost_metrics)
random_forest_df = pd.read_excel(file_path_random_forest_metrics)
logistic_regression_df = pd.read_excel(file_path_logistic_regression_metrics)
lightgbm_df = pd.read_excel(file_path_lightgbm_metrics)
customnn_df = pd.read_excel(file_path_custom_neural_network)


# ---------- COMBINE ALL MODELS ----------
combined_df = pd.concat(
    [
        xgboost_df,
        random_forest_df,
        logistic_regression_df,
        lightgbm_df,
        customnn_df
    ],
    ignore_index=True,
    sort=False
)


# ---------- REORDER BY PERFORMANCE ----------
combined_df.sort_values(
    by='F1',
    ascending=False,
    inplace=True
)


# ---------- SAVE COMBINED EXCEL FILE ----------
output_path = "Dengue_All_Models_Comparison.xlsx"

combined_df.to_excel(
    output_path,
    index=False,
    sheet_name='Model Comparison'
)


# ---------- DISPLAY RESULTS ----------
# print("\nAll Models Comparison:\n")
# print(combined_df.to_string(index=False))

# print(f"\nCombined Excel file saved: {output_path}")