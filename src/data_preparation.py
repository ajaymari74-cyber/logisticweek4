"""
Data Preparation Module for Logistics Predictive Modeling
Handles data ingestion, validation, leakage prevention, and train/test splitting.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Delivery_Time_Days"

# Columns to strictly exclude to prevent Data Leakage or non-generalizable IDs
LEAKAGE_AND_ID_COLUMNS = [
    "Order_ID",
    "Delivery_Status",         # Observed post-delivery
    "Customer_Rating",         # Post-delivery customer feedback
    "Shipping_Delay_Days",     # Calculated using actual delivery time
    "Is_Delayed",              # Binary flag from actual delivery time
    "Speed_Index_KMPD",        # Distance / Delivery_Time_Days (Direct Leakage)
    "Norm_Delivery_Time_Days", # Normalized target (Direct Leakage)
    # Pre-encoded or pre-normalized columns from week 2 (we will encode fresh in pipeline)
    "Enc_Shipping_Mode_Express Air",
    "Enc_Shipping_Mode_Ground Freight",
    "Enc_Shipping_Mode_Same-Day Courier",
    "Enc_Shipping_Mode_Standard Delivery",
    "Enc_Customer_Segment_Consumer",
    "Enc_Customer_Segment_Corporate",
    "Enc_Customer_Segment_Home Office",
    "Enc_Customer_Segment_Small Business",
    "Enc_Region_Central",
    "Enc_Region_East",
    "Enc_Region_North",
    "Enc_Region_South",
    "Enc_Region_West",
    "Norm_Quantity",
    "Norm_Sales_USD",
    "Norm_Shipping_Cost_USD",
    "Norm_Distance_KM",
    "Norm_Cost_Per_Unit",
    "Norm_Cost_Per_KM",
    # Post-dispatch date columns
    "Shipping_Date"            # Dispatch date; we retain Order_Date and fulfillment days
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file and perform initial validation.
    
    Args:
        file_path: Absolute or relative path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[Data Prep] Ingested {len(df)} records with {df.shape[1]} columns from {file_path}")
    return df


def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data completeness, null counts, duplicate records, and target distribution.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dict[str, Any]: Integrity metrics and summary.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    duplicates = df.duplicated().sum()
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset!")
        
    target_stats = {
        "count": int(df[TARGET_COLUMN].count()),
        "mean": float(df[TARGET_COLUMN].mean()),
        "std": float(df[TARGET_COLUMN].std()),
        "min": float(df[TARGET_COLUMN].min()),
        "q25": float(df[TARGET_COLUMN].quantile(0.25)),
        "median": float(df[TARGET_COLUMN].median()),
        "q75": float(df[TARGET_COLUMN].quantile(0.75)),
        "max": float(df[TARGET_COLUMN].max())
    }
    
    report = {
        "total_rows": total_rows,
        "total_columns": df.shape[1],
        "total_missing_values": int(total_missing),
        "duplicate_rows": int(duplicates),
        "target_summary": target_stats
    }
    
    print(f"[Data Prep] Data Integrity: {total_rows} rows, {total_missing} missing values, {duplicates} duplicates.")
    print(f"[Data Prep] Target '{TARGET_COLUMN}' Mean: {target_stats['mean']:.2f} days (Range: {target_stats['min']:.1f} - {target_stats['max']:.1f} days).")
    return report


def isolate_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Isolate predictive features and the target variable while removing data leakage columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, List[str]]: X (features), y (target), excluded_columns.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
    
    y = df[TARGET_COLUMN].copy()
    
    # Identify columns to drop (leakage + target itself)
    cols_to_drop = [col for col in LEAKAGE_AND_ID_COLUMNS if col in df.columns]
    if TARGET_COLUMN not in cols_to_drop:
        cols_to_drop.append(TARGET_COLUMN)
        
    X = df.drop(columns=cols_to_drop).copy()
    
    print(f"[Data Prep] Isolated target '{TARGET_COLUMN}'. Excluded {len(cols_to_drop)} leakage/ID columns.")
    print(f"[Data Prep] Retained {X.shape[1]} predictive raw features: {list(X.columns)}")
    
    return X, y, cols_to_drop


def split_train_test(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42,
    chronological: bool = False,
    date_col: str = "Order_Date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and target vector into training and testing partitions.
    
    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Fraction of samples assigned to the test partition (default: 0.20).
        random_state: Seed for reproducible random splitting.
        chronological: If True, split based on order date to simulate temporal out-of-time evaluation.
        date_col: Column name containing dates for temporal splitting.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    if chronological and date_col in X.columns:
        print("[Data Prep] Performing chronological train/test split based on Order_Date...")
        sorted_indices = pd.to_datetime(X[date_col]).sort_values().index
        split_idx = int(len(sorted_indices) * (1 - test_size))
        
        train_idx = sorted_indices[:split_idx]
        test_idx = sorted_indices[split_idx:]
        
        X_train, X_test = X.loc[train_idx].copy(), X.loc[test_idx].copy()
        y_train, y_test = y.loc[train_idx].copy(), y.loc[test_idx].copy()
    else:
        print(f"[Data Prep] Performing stratified/random train/test split (test_size={test_size}, random_state={random_state})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    print(f"[Data Prep] Train partition: {X_train.shape[0]} samples (80.0%) | Test partition: {X_test.shape[0]} samples (20.0%)")
    return X_train, X_test, y_train, y_test
