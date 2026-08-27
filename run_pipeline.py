"""
End-to-End Machine Learning and Optimization Pipeline Orchestrator
Week 4: Predictive Modeling and Optimization in Logistics Systems
"""

import os
import json
import numpy as np
import pandas as pd

from src.data_preparation import (
    load_dataset,
    validate_data_integrity,
    isolate_features_and_target,
    split_train_test
)
from src.feature_engineering import (
    engineer_features,
    identify_feature_types,
    build_preprocessor_pipeline,
    get_feature_names_after_preprocessing
)
from src.models import (
    build_candidate_pipelines,
    train_candidate_models,
    tune_best_model
)
from src.evaluation import (
    compute_regression_metrics,
    evaluate_and_compare_models,
    extract_feature_importances,
    perform_cross_validation
)
from src.visualization import (
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_residual_distribution,
    plot_model_comparison,
    plot_cross_validation_stability,
    plot_feature_importance,
    plot_tuning_impact,
    plot_optimization_tradeoff,
    plot_regional_optimization_summary
)
from src.optimization import (
    extract_empirical_logistics_parameters,
    solve_resource_allocation_lp,
    compare_baseline_vs_optimized
)


def run_full_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "processed", "logistics_cleaned.csv")
    fig_dir = os.path.join(project_root, "outputs", "figures")
    metrics_dir = os.path.join(project_root, "outputs", "metrics")
    pred_dir = os.path.join(project_root, "outputs", "predictions")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    print("=" * 80)
    print("STARTING WEEK 4 LOGISTICS PREDICTIVE MODELING & OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # 1. Ingestion & Validation
    df = load_dataset(data_path)
    integrity_report = validate_data_integrity(df)
    
    # 2. Target Isolation & Anti-Leakage Feature Extraction
    X_raw, y, dropped_cols = isolate_features_and_target(df)
    
    # 3. Train/Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = split_train_test(
        X_raw, y, test_size=0.20, random_state=42
    )
    
    # 4. Feature Engineering
    X_train_eng = engineer_features(X_train_raw)
    X_test_eng = engineer_features(X_test_raw)
    
    num_cols, cat_cols = identify_feature_types(X_train_eng)
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    
    # Fit preprocessor on train to recover feature names
    preprocessor.fit(X_train_eng)
    transformed_feature_names = get_feature_names_after_preprocessing(preprocessor, num_cols, cat_cols)
    print(f"[Pipeline] Preprocessing expands input into {len(transformed_feature_names)} engineered features.")
    
    # 5. Build & Train Candidate Models
    candidate_pipes = build_candidate_pipelines(preprocessor, random_state=42)
    fitted_pipes, train_times = train_candidate_models(candidate_pipes, X_train_eng, y_train)
    
    # 6. Evaluate and Compare Candidate Models
    comp_df, predictions_dict, diagnostics_dict = evaluate_and_compare_models(
        fitted_pipes, train_times, X_train_eng, y_train, X_test_eng, y_test, cv=5
    )
    comp_df.to_csv(os.path.join(metrics_dir, "model_comparison.csv"), index=False)
    
    # 7. Identify Top Performing Model for Hyperparameter Tuning
    best_candidate_name = comp_df.iloc[0]["Model"]
    print(f"\n[Pipeline] Top Candidate Model Selected: '{best_candidate_name}'")
    
    tune_target = "Random Forest" if "Random Forest" in best_candidate_name or "Tree" in best_candidate_name else "Gradient Boosting"
    tuned_pipe, tuning_report, tuning_duration = tune_best_model(
        preprocessor, X_train_eng, y_train, model_type=tune_target, cv=5, random_state=42
    )
    
    # Evaluate Tuned Model on Test Partition
    y_pred_tuned = tuned_pipe.predict(X_test_eng)
    tuned_metrics = compute_regression_metrics(y_test.values, y_pred_tuned)
    tuned_cv = perform_cross_validation(tuned_pipe, X_train_eng, y_train, cv=5)
    
    tuned_record = {
        "Rank": 1,
        "Model": f"Tuned {tune_target} (Best)",
        "Test_MAE (Days)": tuned_metrics["MAE"],
        "Test_RMSE (Days)": tuned_metrics["RMSE"],
        "Test_R2": tuned_metrics["R2"],
        "Test_MAPE (%)": tuned_metrics["MAPE_Percent"],
        "CV_Mean_MAE": tuned_cv["CV_Mean_MAE"],
        "CV_Std_MAE": tuned_cv["CV_Std_MAE"],
        "CV_Mean_R2": tuned_cv["CV_Mean_R2"],
        "Training_Time (s)": round(tuning_duration, 4)
    }
    
    # Append Tuned Best Model to Final Summary Table
    final_comp_df = pd.concat([pd.DataFrame([tuned_record]), comp_df], ignore_index=True)
    # Re-rank
    final_comp_df = final_comp_df.sort_values(by=["Test_MAE (Days)", "Test_RMSE (Days)"], ascending=[True, True]).reset_index(drop=True)
    final_comp_df["Rank"] = range(1, len(final_comp_df) + 1)
    final_comp_df.to_csv(os.path.join(metrics_dir, "final_model_benchmarks.csv"), index=False)
    
    # Save tuning metadata
    tuning_export = {
        "target_model": tune_target,
        "best_hyperparameters": tuning_report["best_params"],
        "best_cv_mae": tuning_report["best_cv_mae"],
        "test_metrics": tuned_metrics,
        "cv_metrics": tuned_cv,
        "tuning_duration_seconds": tuning_report["tuning_duration_seconds"]
    }
    with open(os.path.join(metrics_dir, "hyperparameter_tuning.json"), "w") as f:
        json.dump(tuning_export, f, indent=4)
        
    # 8. Feature Importance
    importance_df = extract_feature_importances(tuned_pipe, transformed_feature_names)
    importance_df.to_csv(os.path.join(metrics_dir, "feature_importance.csv"), index=False)
    print("\n[Pipeline] Top 10 Most Influential Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # 9. Save Predictions & Demonstration Cases
    test_pred_df = X_test_raw.copy()
    test_pred_df["Actual_Delivery_Days"] = y_test.values
    test_pred_df["Predicted_Delivery_Days"] = y_pred_tuned.round(2)
    test_pred_df["Absolute_Error"] = (np.abs(y_test.values - y_pred_tuned)).round(2)
    test_pred_df["Percentage_Error"] = (np.abs(y_test.values - y_pred_tuned) / y_test.values * 100.0).round(2)
    test_pred_df.to_csv(os.path.join(pred_dir, "test_set_predictions.csv"), index=False)
    
    # Extract 5 Diverse Real Demonstration Cases
    demo_cases = test_pred_df.head(5)[
        ["Shipping_Mode", "Distance_KM", "Quantity", "Product_Category", "Region", "Warehouse_Code", "Actual_Delivery_Days", "Predicted_Delivery_Days", "Absolute_Error"]
    ]
    demo_cases.to_csv(os.path.join(pred_dir, "demonstration_cases.csv"), index=False)
    
    # 10. Generate Publication-Quality Visualizations
    print("\n[Pipeline] Generating publication-quality charts in outputs/figures/...")
    residuals_tuned = y_test.values - y_pred_tuned
    
    plot_actual_vs_predicted(
        y_test.values, y_pred_tuned, f"Tuned {tune_target}",
        os.path.join(fig_dir, "01_actual_vs_predicted.png"), tuned_metrics
    )
    
    plot_residual_analysis(
        y_pred_tuned, residuals_tuned, f"Tuned {tune_target}",
        os.path.join(fig_dir, "02_residual_analysis.png")
    )
    
    plot_residual_distribution(
        residuals_tuned, f"Tuned {tune_target}",
        os.path.join(fig_dir, "03_residual_distribution.png")
    )
    
    plot_model_comparison(
        final_comp_df, os.path.join(fig_dir, "04_model_comparison.png")
    )
    
    plot_cross_validation_stability(
        final_comp_df, os.path.join(fig_dir, "05_cross_validation_stability.png")
    )
    
    plot_feature_importance(
        importance_df, f"Tuned {tune_target}",
        os.path.join(fig_dir, "06_feature_importance.png"), top_n=12
    )
    
    # Untuned vs Tuned
    untuned_metrics_base = diagnostics_dict[best_candidate_name]["metrics"]
    plot_tuning_impact(
        untuned_metrics_base, tuned_metrics, tune_target,
        os.path.join(fig_dir, "07_hyperparameter_tuning_impact.png")
    )
    
    # 11. Operational Optimization Formulation and Solving
    print("\n[Pipeline] Formulating and solving Logistics Resource & Mode Allocation LP...")
    opt_params = extract_empirical_logistics_parameters(df, best_model=tuned_pipe)
    opt_results = solve_resource_allocation_lp(opt_params)
    opt_summary_df, regional_opt_df = compare_baseline_vs_optimized(df, opt_params, opt_results)
    
    opt_summary_df.to_csv(os.path.join(metrics_dir, "optimization_summary.csv"), index=False)
    regional_opt_df.to_csv(os.path.join(metrics_dir, "regional_optimization.csv"), index=False)
    
    # Visualizations for Optimization
    plot_optimization_tradeoff(
        opt_summary_df, os.path.join(fig_dir, "08_optimization_cost_time_tradeoff.png")
    )
    plot_regional_optimization_summary(
        regional_opt_df, os.path.join(fig_dir, "09_regional_optimization_summary.png")
    )
    
    print("\n" + "=" * 80)
    print("WEEK 4 PIPELINE COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
