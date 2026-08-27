"""
Feature Engineering Module for Logistics Predictive Modeling
Creates domain-specific logistics features, temporal attributes, and Scikit-Learn preprocessing pipelines.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific logistics features and temporal attributes.
    
    Args:
        df: Input feature DataFrame (without target or leakage columns).
        
    Returns:
        pd.DataFrame: Feature-enriched DataFrame.
    """
    X_feat = df.copy()
    
    # 1. Temporal Engineering from Order_Date (if present)
    if "Order_Date" in X_feat.columns:
        dates = pd.to_datetime(X_feat["Order_Date"], errors="coerce")
        X_feat["Order_Month"] = dates.dt.month.fillna(1).astype(int)
        X_feat["Order_DayOfWeek"] = dates.dt.dayofweek.fillna(0).astype(int)
        X_feat["Is_Weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        X_feat["Order_Day"] = dates.dt.day.fillna(1).astype(int)
        # Drop raw string date column
        X_feat = X_feat.drop(columns=["Order_Date"])
        
    # 2. Distance Categorization
    if "Distance_KM" in X_feat.columns:
        X_feat["Distance_Category"] = pd.cut(
            X_feat["Distance_KM"],
            bins=[-np.inf, 500, 1200, np.inf],
            labels=["Short", "Medium", "Long"]
        ).astype(str)
        
    # 3. Unit Economics & Ratios
    if "Shipping_Cost_USD" in X_feat.columns and "Quantity" in X_feat.columns:
        # Safe division to prevent div-by-zero
        safe_qty = X_feat["Quantity"].replace(0, 1)
        X_feat["Cost_Per_Unit"] = (X_feat["Shipping_Cost_USD"] / safe_qty).round(4)
        
    if "Shipping_Cost_USD" in X_feat.columns and "Distance_KM" in X_feat.columns:
        safe_dist = X_feat["Distance_KM"].replace(0, 1.0)
        X_feat["Cost_Per_KM"] = (X_feat["Shipping_Cost_USD"] / safe_dist).round(4)
        
    if "Sales_USD" in X_feat.columns and "Quantity" in X_feat.columns:
        safe_qty = X_feat["Quantity"].replace(0, 1)
        X_feat["Value_Density"] = (X_feat["Sales_USD"] / safe_qty).round(2)
        
    # 4. Regional Proximity Interaction
    # E.g. WH-Central matching Central region
    if "Warehouse_Code" in X_feat.columns and "Region" in X_feat.columns:
        wh_region = X_feat["Warehouse_Code"].str.replace("WH-", "", regex=False)
        X_feat["Is_Local_Dispatch"] = (wh_region == X_feat["Region"]).astype(int)
        
    return X_feat


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical column names from the engineered DataFrame.
    
    Args:
        X: Feature DataFrame.
        
    Returns:
        Tuple[List[str], List[str]]: numerical_columns, categorical_columns.
    """
    numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"[Feature Eng] Identified {len(numerical_cols)} numerical features: {numerical_cols}")
    print(f"[Feature Eng] Identified {len(categorical_cols)} categorical features: {categorical_cols}")
    return numerical_cols, categorical_cols


def build_preprocessor_pipeline(
    numerical_cols: List[str], 
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Build a robust Scikit-Learn ColumnTransformer pipeline with imputation, scaling, and one-hot encoding.
    
    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer, 
    numerical_cols: List[str], 
    categorical_cols: List[str]
) -> List[str]:
    """
    Recover human-readable feature names post ColumnTransformer preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        numerical_cols: Input numerical columns.
        categorical_cols: Input categorical columns.
        
    Returns:
        List[str]: Transformed feature names.
    """
    feature_names = []
    
    # Numerical names (unchanged)
    feature_names.extend(numerical_cols)
    
    # Categorical one-hot names
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)
    except Exception as e:
        print(f"[Warning] Could not extract categorical feature names dynamically: {e}")
        feature_names.extend([f"cat_feat_{i}" for i in range(len(categorical_cols))])
        
    return feature_names
