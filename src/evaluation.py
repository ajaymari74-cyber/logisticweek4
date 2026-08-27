"""
Model Evaluation and Diagnostics Module for Logistics Predictive Modeling
Computes regression metrics (MAE, RMSE, R2, MAPE), performs 5-Fold Cross-Validation,
ranks model performance, and extracts feature importances.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate core regression metrics: MAE, RMSE, R2, MAPE, and Max Error.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Model predicted target values.
        
    Returns:
        Dict[str, float]: Calculated metrics dictionary.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Safe MAPE calculation
    safe_true = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_true)) * 100.0
    max_err = np.max(np.abs(y_true - y_pred))
    
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE_Percent": round(float(mape), 2),
        "Max_Error": round(float(max_err), 4)
    }


def perform_cross_validation(
    pipeline: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv: int = 5
) -> Dict[str, float]:
    """
    Run 5-Fold Cross-Validation on the training set to evaluate stability and generalization.
    
    Args:
        pipeline: Scikit-learn Pipeline.
        X_train: Training features.
        y_train: Training target.
        cv: Number of folds (default: 5).
        
    Returns:
        Dict[str, float]: CV mean MAE, CV std MAE, CV mean R2.
    """
    # Negative MAE scoring (standard scikit-learn convention)
    neg_mae_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    mae_scores = -neg_mae_scores
    
    r2_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
    )
    
    return {
        "CV_Mean_MAE": round(float(np.mean(mae_scores)), 4),
        "CV_Std_MAE": round(float(np.std(mae_scores)), 4),
        "CV_Mean_R2": round(float(np.mean(r2_scores)), 4),
        "CV_Std_R2": round(float(np.std(r2_scores)), 4),
        "All_CV_MAE_Folds": [round(float(s), 4) for s in mae_scores]
    }


def evaluate_and_compare_models(
    fitted_pipelines: Dict[str, Pipeline],
    training_times: Dict[str, float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """
    Evaluate all candidate pipelines on the test set, perform cross-validation, and rank models.
    
    Args:
        fitted_pipelines: Dictionary of fitted models/pipelines.
        training_times: Dictionary of training durations in seconds.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        cv: Number of CV folds.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
            Comparison DataFrame, Predictions dict, Detailed diagnostics dict.
    """
    comparison_records = []
    predictions_dict = {}
    diagnostics_dict = {}
    
    print("\n" + "=" * 80)
    print(f"{'MODEL EVALUATION & BENCHMARKING':^80}")
    print("=" * 80)
    
    for name, pipe in fitted_pipelines.items():
        # Predict on Test partition
        y_pred = pipe.predict(X_test)
        predictions_dict[name] = y_pred
        
        # Test Metrics
        metrics = compute_regression_metrics(y_test.values, y_pred)
        
        # Cross-Validation on Train partition
        cv_res = perform_cross_validation(pipe, X_train, y_train, cv=cv)
        
        # Diagnostics
        residuals = y_test.values - y_pred
        diagnostics_dict[name] = {
            "metrics": metrics,
            "cv": cv_res,
            "residuals": residuals,
            "res_mean": float(np.mean(residuals)),
            "res_std": float(np.std(residuals))
        }
        
        train_time = training_times.get(name, 0.0)
        
        record = {
            "Model": name,
            "Test_MAE (Days)": metrics["MAE"],
            "Test_RMSE (Days)": metrics["RMSE"],
            "Test_R2": metrics["R2"],
            "Test_MAPE (%)": metrics["MAPE_Percent"],
            "CV_Mean_MAE": cv_res["CV_Mean_MAE"],
            "CV_Std_MAE": cv_res["CV_Std_MAE"],
            "CV_Mean_R2": cv_res["CV_Mean_R2"],
            "Training_Time (s)": round(train_time, 4)
        }
        comparison_records.append(record)
        
    comp_df = pd.DataFrame(comparison_records)
    # Rank models primarily by Test_MAE ascending, then Test_R2 descending
    comp_df = comp_df.sort_values(by=["Test_MAE (Days)", "Test_RMSE (Days)"], ascending=[True, True]).reset_index(drop=True)
    comp_df["Rank"] = range(1, len(comp_df) + 1)
    
    # Reorder columns
    cols_order = ["Rank", "Model", "Test_MAE (Days)", "Test_RMSE (Days)", "Test_R2", "Test_MAPE (%)", "CV_Mean_MAE", "CV_Std_MAE", "CV_Mean_R2", "Training_Time (s)"]
    comp_df = comp_df[cols_order]
    
    print(comp_df.to_string(index=False))
    print("=" * 80)
    
    return comp_df, predictions_dict, diagnostics_dict


def extract_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance scores or coefficients from a fitted model pipeline.
    
    Args:
        pipeline: Fitted Scikit-Learn Pipeline.
        feature_names: List of preprocessed feature names.
        
    Returns:
        pd.DataFrame: Sorted feature importances DataFrame.
    """
    model = pipeline.named_steps["model"]
    
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
        metric_name = "Importance_Score"
    elif hasattr(model, "coef_"):
        raw_importances = np.abs(model.coef_)
        metric_name = "Absolute_Coefficient"
    else:
        print("[Warning] Model does not provide feature importances or coefficients.")
        return pd.DataFrame()
        
    if len(raw_importances) != len(feature_names):
        # Truncate or pad if feature count differs slightly due to encoding
        min_len = min(len(raw_importances), len(feature_names))
        raw_importances = raw_importances[:min_len]
        feature_names = feature_names[:min_len]
        
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        metric_name: raw_importances
    })
    
    # Calculate percentage contribution
    total_val = imp_df[metric_name].sum()
    if total_val > 0:
        imp_df["Relative_Contribution_Percent"] = (imp_df[metric_name] / total_val * 100.0).round(2)
    else:
        imp_df["Relative_Contribution_Percent"] = 0.0
        
    imp_df = imp_df.sort_values(by=metric_name, ascending=False).reset_index(drop=True)
    return imp_df
