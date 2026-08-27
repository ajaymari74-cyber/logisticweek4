"""
Machine Learning Models Module for Logistics Predictive Modeling
Defines baseline, linear, tree, and ensemble regression models and hyperparameter tuning workflows.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def build_candidate_pipelines(preprocessor: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Construct full Scikit-Learn pipelines pairing preprocessing with regression algorithms.
    
    Args:
        preprocessor: Configured ColumnTransformer.
        random_state: Random state for reproducibility.
        
    Returns:
        Dict[str, Pipeline]: Dictionary of candidate pipelines.
    """
    pipelines = {
        "Baseline (Mean)": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=random_state))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=random_state, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state))
        ])
    }
    
    return pipelines


def train_candidate_models(
    pipelines: Dict[str, Pipeline], 
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[Dict[str, Pipeline], Dict[str, float]]:
    """
    Fit each candidate pipeline on the training partition and record execution time.
    
    Args:
        pipelines: Dictionary of candidate pipelines.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        
    Returns:
        Tuple[Dict[str, Pipeline], Dict[str, float]]: Fitted pipelines and training durations in seconds.
    """
    fitted_pipelines = {}
    training_times = {}
    
    for name, pipe in pipelines.items():
        start_t = time.perf_counter()
        pipe.fit(X_train, y_train)
        duration = time.perf_counter() - start_t
        
        fitted_pipelines[name] = pipe
        training_times[name] = duration
        print(f"[Model Training] {name:20s} fitted in {duration:.4f}s")
        
    return fitted_pipelines, training_times


def tune_best_model(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "Random Forest",
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Perform GridSearchCV hyperparameter tuning on the selected top ensemble architecture.
    
    Args:
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        model_type: Target model to tune ("Random Forest" or "Gradient Boosting").
        cv: Number of cross-validation folds.
        random_state: Random state seed.
        
    Returns:
        Tuple[Pipeline, Dict[str, Any], float]: Best fitted pipeline, best parameters, and search duration.
    """
    print(f"\n[Hyperparameter Tuning] Initiating GridSearchCV on {model_type} ({cv}-Fold CV)...")
    
    if model_type == "Gradient Boosting":
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=random_state))
        ])
        param_grid = {
            "model__n_estimators": [75, 120, 180],
            "model__learning_rate": [0.03, 0.08, 0.15],
            "model__max_depth": [3, 4, 5],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 3]
        }
    else:  # Default: Random Forest
        base_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1))
        ])
        param_grid = {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_split": [2, 5, 8],
            "model__min_samples_leaf": [1, 2, 4]
        }
        
    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    start_t = time.perf_counter()
    grid_search.fit(X_train, y_train)
    tuning_duration = time.perf_counter() - start_t
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_mae = -grid_search.best_score_
    
    print(f"[Hyperparameter Tuning] Completed in {tuning_duration:.2f}s")
    print(f"[Hyperparameter Tuning] Best CV MAE: {best_cv_mae:.4f} days")
    print(f"[Hyperparameter Tuning] Optimal Parameters: {best_params}")
    
    return best_pipeline, {
        "best_params": best_params,
        "best_cv_mae": float(best_cv_mae),
        "total_combinations_tested": len(grid_search.cv_results_["params"]),
        "tuning_duration_seconds": float(tuning_duration)
    }, tuning_duration
