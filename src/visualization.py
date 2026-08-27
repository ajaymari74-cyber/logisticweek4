"""
Visualization Module for Logistics Predictive Modeling & Optimization
Generates publication-quality charts, diagnostic plots, and optimization visuals.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List


# Set global aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Custom Brand Palette
NAVY = "#1E3A8A"
TEAL = "#0D9488"
SLATE = "#1E293B"
AMBER = "#D97706"
ROSE = "#E11D48"
INDIGO = "#4F46E5"
EMERALD = "#059669"
GRAY_BG = "#F8FAFC"
BORDER_GRAY = "#CBD5E1"


def plot_actual_vs_predicted(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    model_name: str,
    output_path: str,
    metrics: Dict[str, float]
) -> None:
    """
    Generate an Actual vs. Predicted scatter plot with identity line and error density.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Scatter plot
    scatter = ax.scatter(
        y_true, y_pred, 
        alpha=0.65, 
        color=TEAL, 
        edgecolors="white", 
        linewidth=0.6, 
        s=60, 
        label="Test Sample Predictions"
    )
    
    # 45-degree reference line
    min_val = min(np.min(y_true), np.min(y_pred)) - 0.5
    max_val = max(np.max(y_true), np.max(y_pred)) + 0.5
    ax.plot([min_val, max_val], [min_val, max_val], color=ROSE, linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
    
    # Linear fit line
    sns.regplot(
        x=y_true, y=y_pred, ax=ax, scatter=False, 
        color=NAVY, line_kws={"linewidth": 1.5, "linestyle": "-.", "label": "Empirical Regression Trend"}
    )
    
    # Text annotation box with key metrics
    annotation_text = (
        f"Model: {model_name}\n"
        f"R² Score: {metrics.get('R2', 0.0):.4f}\n"
        f"MAE: {metrics.get('MAE', 0.0):.4f} days\n"
        f"RMSE: {metrics.get('RMSE', 0.0):.4f} days"
    )
    ax.text(
        0.05, 0.93, annotation_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=GRAY_BG, edgecolor=BORDER_GRAY, alpha=0.95)
    )
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual Delivery Time (Days)", fontweight="bold")
    ax.set_ylabel("Predicted Delivery Time (Days)", fontweight="bold")
    ax.set_title(f"Actual vs. Predicted Delivery Time — {model_name}", pad=14)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_residual_analysis(
    y_pred: np.ndarray, 
    residuals: np.ndarray, 
    model_name: str,
    output_path: str
) -> None:
    """
    Generate residual scatter plot vs fitted values to assess homoscedasticity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(
        y_pred, residuals, 
        alpha=0.65, 
        color=INDIGO, 
        edgecolors="white", 
        linewidth=0.6, 
        s=55, 
        label=r"Residuals ($e_i = y_i - \hat{y}_i$)"
    )
    
    ax.axhline(0, color=ROSE, linestyle="--", linewidth=1.8, label="Zero Error Baseline")
    
    # Polynomial trend line without requiring statsmodels
    sns.regplot(
        x=y_pred, y=residuals, ax=ax, scatter=False, order=2,
        color=AMBER, line_kws={"linewidth": 2, "label": "Residual Polynomial Trend"}
    )
    
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    ax.axhline(res_mean + 2 * res_std, color="gray", linestyle=":", alpha=0.7, label=r"$\pm 2\sigma$ Boundary")
    ax.axhline(res_mean - 2 * res_std, color="gray", linestyle=":", alpha=0.7)
    
    ax.set_xlabel("Fitted / Predicted Delivery Time (Days)", fontweight="bold")
    ax.set_ylabel("Residual Error (Actual - Predicted, Days)", fontweight="bold")
    ax.set_title(f"Residual Diagnostic Plot — {model_name}", pad=14)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_residual_distribution(
    residuals: np.ndarray, 
    model_name: str,
    output_path: str
) -> None:
    """
    Generate a 2-panel residual distribution plot: Histogram with KDE and Q-Q Plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # 1. Histogram + Normal Fit
    sns.histplot(
        residuals, kde=True, ax=axes[0], color=TEAL, stat="density",
        bins=25, edgecolor="white", alpha=0.6
    )
    # Overlay theoretical standard normal curve
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = axes[0].get_xlim()
    x_grid = np.linspace(xmin, xmax, 100)
    p_grid = stats.norm.pdf(x_grid, mu, std)
    axes[0].plot(x_grid, p_grid, color=ROSE, linewidth=2, linestyle="--", label=rf"Normal Curve ($\mu={mu:.2f}, \sigma={std:.2f}$)")
    
    axes[0].set_title(f"Residual Error Density & Normality", pad=10)
    axes[0].set_xlabel("Residual Error (Days)", fontweight="bold")
    axes[0].set_ylabel("Density", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    
    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor(INDIGO)
    axes[1].get_lines()[0].set_markeredgecolor("white")
    axes[1].get_lines()[0].set_alpha(0.7)
    axes[1].get_lines()[0].set_markersize(6)
    axes[1].get_lines()[1].set_color(ROSE)
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title("Normal Q-Q Probability Plot", pad=10)
    axes[1].set_xlabel("Theoretical Quantiles", fontweight="bold")
    axes[1].set_ylabel("Sample Quantiles", fontweight="bold")
    
    fig.suptitle(f"Residual Error Normality Diagnostics — {model_name}", y=1.02, fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame, 
    output_path: str
) -> None:
    """
    Generate a 3-panel comparative bar chart for MAE, RMSE, and R2 across candidate models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    models = comparison_df["Model"].tolist()
    colors = [NAVY if i == 0 else TEAL if i == 1 else INDIGO for i in range(len(models))]
    
    # 1. MAE
    bars1 = axes[0].bar(models, comparison_df["Test_MAE (Days)"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[0].set_title("Test Mean Absolute Error (MAE ↓)", pad=10)
    axes[0].set_ylabel("MAE (Days)", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # 2. RMSE
    bars2 = axes[1].bar(models, comparison_df["Test_RMSE (Days)"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].set_title("Test Root Mean Squared Error (RMSE ↓)", pad=10)
    axes[1].set_ylabel("RMSE (Days)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # 3. R2 Score
    bars3 = axes[2].bar(models, comparison_df["Test_R2"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[2].set_title("Test Coefficient of Determination ($R^2$ ↑)", pad=10)
    axes[2].set_ylabel("R² Score", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=30)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, max(0.01, yval + 0.01), f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    fig.suptitle("Predictive Model Performance Comparison (Test Partition)", y=1.03, fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_cross_validation_stability(
    comparison_df: pd.DataFrame, 
    output_path: str
) -> None:
    """
    Generate a bar chart with error bars showing 5-Fold CV Mean MAE and Standard Deviation.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = comparison_df["Model"].tolist()
    mean_maes = comparison_df["CV_Mean_MAE"].tolist()
    std_maes = comparison_df["CV_Std_MAE"].tolist()
    
    bars = ax.bar(
        models, mean_maes, yerr=std_maes, capsize=6,
        color=TEAL, edgecolor=NAVY, linewidth=1, alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": ROSE}
    )
    
    for bar, mean_val, std_val in zip(bars, mean_maes, std_maes):
        ax.text(
            bar.get_x() + bar.get_width()/2.0, 
            mean_val / 2.0, 
            f"{mean_val:.2f}\n±{std_val:.2f}", 
            ha="center", va="center", color="white", fontsize=9, fontweight="bold"
        )
        
    ax.set_ylabel("5-Fold Cross-Validation MAE (Days)", fontweight="bold")
    ax.set_title("5-Fold Cross-Validation Stability & Generalization Error", pad=14)
    ax.tick_params(axis="x", rotation=25)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_feature_importance(
    importance_df: pd.DataFrame, 
    model_name: str,
    output_path: str,
    top_n: int = 12
) -> None:
    """
    Generate horizontal bar chart for top N feature importances.
    """
    if importance_df.empty:
        return
        
    top_df = importance_df.head(top_n).iloc[::-1]  # Reverse for top-down display
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    metric_col = [col for col in ["Importance_Score", "Absolute_Coefficient"] if col in top_df.columns][0]
    
    bars = ax.barh(
        top_df["Feature"], top_df[metric_col],
        color=NAVY, edgecolor=TEAL, linewidth=1, alpha=0.85
    )
    
    # Add percentage labels
    for bar, pct in zip(bars, top_df["Relative_Contribution_Percent"]):
        width = bar.get_width()
        ax.text(
            width + (width * 0.02 + 0.005), bar.get_y() + bar.get_height()/2.0,
            f"{pct:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold", color=SLATE
        )
        
    ax.set_xlabel(f"{metric_col.replace('_', ' ')}", fontweight="bold")
    ax.set_title(f"Top {top_n} Predictive Feature Importances — {model_name}", pad=14)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_tuning_impact(
    untuned_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float],
    model_name: str,
    output_path: str
) -> None:
    """
    Generate a grouped comparison chart between Untuned and Tuned model metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ["MAE (Days ↓)", "RMSE (Days ↓)", "R² Score (↑)"]
    untuned_vals = [untuned_metrics["MAE"], untuned_metrics["RMSE"], untuned_metrics["R2"]]
    tuned_vals = [tuned_metrics["MAE"], tuned_metrics["RMSE"], tuned_metrics["R2"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, untuned_vals, width, label="Default / Baseline Model", color=SLATE, alpha=0.8)
    rects2 = ax.bar(x + width/2, tuned_vals, width, label="GridSearchCV Tuned Model", color=TEAL, alpha=0.9)
    
    # Value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_title(f"Hyperparameter Tuning Impact on {model_name}", pad=14)
    ax.legend(frameon=True, facecolor="white")
    ax.set_ylim(0, max(max(untuned_vals), max(tuned_vals)) * 1.25)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_optimization_tradeoff(
    opt_summary_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate trade-off scatter / Pareto comparison between Total Cost and Delivery Time.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for idx, row in opt_summary_df.iterrows():
        color = ROSE if "Baseline" in row["Scenario"] else EMERALD if "Optimal" in row["Scenario"] else INDIGO
        marker = "o" if "Baseline" in row["Scenario"] else "*" if "Optimal" in row["Scenario"] else "s"
        size = 180 if "Optimal" in row["Scenario"] else 120
        
        ax.scatter(
            row["Avg_Delivery_Time_Days"], row["Total_Logistics_Cost_USD"],
            s=size, color=color, marker=marker, edgecolors="black", linewidth=1.2, zorder=4,
            label=row["Scenario"]
        )
        
        offset_y = 150 if idx % 2 == 0 else -250
        ax.annotate(
            f"{row['Scenario']}\n(${row['Total_Logistics_Cost_USD']:,.0f} | {row['Avg_Delivery_Time_Days']:.2f}d)",
            xy=(row["Avg_Delivery_Time_Days"], row["Total_Logistics_Cost_USD"]),
            xytext=(row["Avg_Delivery_Time_Days"] + 0.08, row["Total_Logistics_Cost_USD"] + offset_y),
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=1),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=BORDER_GRAY, alpha=0.9)
        )
        
    ax.set_xlabel("Average Fleet Delivery Time (Days)", fontweight="bold")
    ax.set_ylabel("Total Operational Logistics Cost (USD)", fontweight="bold")
    ax.set_title("Cost vs. Delivery Lead Time Trade-Off (Optimization Scenarios)", pad=14)
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")


def plot_regional_optimization_summary(
    regional_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate grouped bar chart comparing Before vs After optimization across regions.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    regions = regional_df["Region"].tolist()
    x = np.arange(len(regions))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, regional_df["Baseline_Cost_USD"], width, label="Baseline Empirical Cost ($)", color=SLATE, alpha=0.8)
    rects2 = ax.bar(x + width/2, regional_df["Optimized_Cost_USD"], width, label="Optimized Multi-Mode Cost ($)", color=EMERALD, alpha=0.9)
    
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 300, f"${h:,.0f}", ha="center", va="bottom", fontsize=8.5)
        
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 300, f"${h:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontweight="bold")
    ax.set_ylabel("Logistics Cost (USD)", fontweight="bold")
    ax.set_title("Regional Logistics Cost: Baseline vs. Optimized Model Dispatch", pad=14)
    ax.legend(frameon=True, facecolor="white")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Visualization] Saved: {output_path}")
