# Week 4: Predictive Modeling and Optimization in Logistics Systems

![Python Version](https://img.shields.io/badge/Python-3.14-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.18.1-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)
![Status](https://img.shields.io/badge/Status-Complete%20%26%20Verified-brightgreen.svg)

An end-to-end, production-grade Machine Learning and Operations Research project delivering **Delivery Time Prediction** and **Multi-Region Fleet & Shipping Mode Cost Optimization**.

---

## 📌 Executive Summary & Key Results

* **Predictive Performance**: Evaluated 6 regression architectures under strict **Anti-Leakage** protocols. **Ridge Regression** achieved top performance on the out-of-sample test partition ($N=250$) with a **Mean Absolute Error (MAE) of 0.8930 days** (21.4 hours), a **Root Mean Squared Error (RMSE) of 1.1487 days**, and a **Coefficient of Determination ($R^2$) of 0.7570** (75.7% variance explained), substantially outperforming the naive baseline ($MAE=1.8903$, $R^2=-0.0002$).
* **Cross-Validation Stability**: 5-Fold Cross-Validation confirmed generalization stability with a Mean CV MAE of **0.9540 ± 0.0698 days**.
* **Key Predictive Drivers**: Quoted SLA lead time (`Estimated_Delivery_Days`, 82.16% relative contribution), route distance (`Distance_KM`, 4.91%), and shipping charge (`Shipping_Cost_USD`, 4.63%) are the primary determinants of delivery duration.
* **Operations Research Optimization**: Formulating a multi-region shipping mode Linear Program (LP) in `scipy.optimize` reduced systemic logistics expenditures from **$114,809.33** to **$99,519.46**, capturing **$15,289.87 in net cost savings (13.32% reduction)** while maintaining 100% regional delivery SLA compliance (average delivery time $\le 5.0$ days across all 5 territories).

---

## 🎯 Business Problem & Objectives

Logistics dispatchers frequently face severe operational friction due to static, heuristic-driven delivery scheduling and uncalibrated carrier mode selection:
1. **Unanticipated Transit Delays**: Generate customer dissatisfaction, contract SLA penalties, and emergency carrier re-routing costs.
2. **Suboptimal Expedited Mode Selection**: High-cost express freight modes are over-utilized for non-urgent shipments, while low-cost standard modes suffer delays on long routes.
3. **Regional Capacity Imbalances**: Geographic regions (such as North and West zones) experience compounded fulfillment bottlenecks.

### Project Goals:
1. Ingest the validated Week 2/3 cleaned logistics dataset (1,250 verified transactions) with zero missing values.
2. Formulate domain-specific feature engineering (unit economics, distance categories, local dispatch indicators, temporal calendar variables).
3. Train, benchmark, and cross-validate multiple regression architectures.
4. Perform systematic `GridSearchCV` hyperparameter optimization.
5. Formulate and solve an operational Linear Program using `scipy.optimize` to find the cost-optimal shipping mode dispatch policy.
6. Deliver publication-quality diagnostic visualizations and academic documentation (.md & .docx).

---

## 🛡️ Anti-Leakage Governance

In production machine learning, data leakage occurs when training data includes variables that would not be available at prediction time. The following features are strictly excluded from model training:

| Excluded Leakage Column | Reason for Exclusion |
| :--- | :--- |
| `Shipping_Delay_Days` | Directly derived as $Delivery\_Time - Estimated\_Delivery$; contains target information. |
| `Is_Delayed` | Binary SLA overrun flag observed only after delivery. |
| `Delivery_Status` | Final shipment outcome (`Delivered`, `Delayed`, `Returned`). |
| `Customer_Rating` | Post-delivery satisfaction survey score. |
| `Speed_Index_KMPD` | Calculated as $Distance / Delivery\_Time$ (direct mathematical target leakage). |
| `Norm_Delivery_Time_Days` | Normalized target variable. |
| `Shipping_Date` | Physical dispatch date from warehouse. |

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on the held-out test partition ($N=250$, 20% split) and validated via 5-Fold Cross-Validation:

| Rank | Model Architecture | Test MAE (Days) | Test RMSE (Days) | Test $R^2$ | Test MAPE (%) | CV Mean MAE | CV Std MAE | CV Mean $R^2$ | Training Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Ridge Regression** | **0.8930** | **1.1487** | **0.7570** | **24.24%** | **0.9540** | **0.0698** | **0.6786** | **0.0160** |
| **2** | **Linear Regression** | **0.8943** | **1.1497** | **0.7566** | **24.25%** | **0.9556** | **0.0696** | **0.6777** | **0.0172** |
| **3** | **Random Forest** | **0.9444** | **1.2258** | **0.7233** | **26.63%** | **1.0005** | **0.0694** | **0.6473** | **0.1714** |
| **4** | **Gradient Boosting** | **0.9479** | **1.2243** | **0.7240** | **26.63%** | **1.0114** | **0.0687** | **0.6322** | **0.3277** |
| **5** | **Decision Tree** | **0.9987** | **1.2764** | **0.7000** | **28.19%** | **1.0938** | **0.0548** | **0.5766** | **0.0196** |
| **6** | **Baseline (Mean)** | **1.8903** | **2.3306** | **-0.0002** | **65.80%** | **1.8087** | **0.0704** | **-0.0017** | **0.0146** |

---

## ⚡ Operational Logistics Optimization

### Linear Programming (LP) Formulation:
$$\min Z = \sum_{r \in \text{Regions}} \sum_{m \in \text{Modes}} C_{r, m} \cdot X_{r, m}$$

**Subject to:**
1. **Regional Demand**: $\sum_{m} X_{r, m} = D_r \quad \forall r \in \{\text{Central, East, North, South, West}\}$
2. **Mode Fleet Capacities**: $\sum_{r} X_{r, m} \le \text{Cap}_m \quad \forall m \in \{\text{Express Air, Ground Freight, Same-Day Courier, Standard Delivery}\}$
3. **Regional Lead Time SLA ($\le 5.0$ Days)**: $\frac{1}{D_r} \sum_{m} T_{r, m} \cdot X_{r, m} \le 5.0 \quad \forall r$
4. **Non-negativity**: $X_{r, m} \ge 0$

### Optimization Performance Summary:
| Operational Metric | Baseline (Empirical Dispatch) | ML-Driven Optimal Dispatch | Net Savings / Impact |
| :--- | :---: | :---: | :---: |
| **Total Logistics Expenditure** | **$114,809.33** | **$99,519.46** | **+$15,289.87 (+13.32%)** |
| **Average Delivery Time** | **5.00 days** | **5.00 days** | **Maintained at Target SLA** |
| **Total Demand Fulfilled** | **1,250 orders** | **1,250 orders** | **100% Demand Satisfaction** |
| **SLA Overruns / Violations** | Uncontrolled | **0 Overruns** | **100% SLA Compliance** |

### Regional Breakdown:
* **North Region**: Baseline Cost: $30,140.32 → Optimized Cost: $23,941.87 (**$6,198.45 / 20.57% savings**)
* **West Region**: Baseline Cost: $26,397.28 → Optimized Cost: $22,957.56 (**$3,439.72 / 13.03% savings**)
* **East Region**: Baseline Cost: $18,866.49 → Optimized Cost: $16,519.48 (**$2,347.01 / 12.44% savings**)
* **Central Region**: Baseline Cost: $15,974.29 → Optimized Cost: $13,904.89 (**$2,069.40 / 12.95% savings**)
* **South Region**: Baseline Cost: $23,430.95 → Optimized Cost: $22,196.52 (**$1,234.43 / 5.27% savings**)

---

## 🗂️ Project Directory Architecture

```text
week4-logistics-predictive-modeling/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py
├── convert_docs_to_word.py
├── build_and_execute_notebook.py
├── data/
│   ├── raw/
│   │   └── logistics_data.csv
│   └── processed/
│       └── logistics_cleaned.csv
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── optimization.py
├── notebooks/
│   └── week4_predictive_modeling.ipynb
├── outputs/
│   ├── figures/
│   │   ├── 01_actual_vs_predicted.png
│   │   ├── 02_residual_analysis.png
│   │   ├── 03_residual_distribution.png
│   │   ├── 04_model_comparison.png
│   │   ├── 05_cross_validation_stability.png
│   │   ├── 06_feature_importance.png
│   │   ├── 07_hyperparameter_tuning_impact.png
│   │   ├── 08_optimization_cost_time_tradeoff.png
│   │   └── 09_regional_optimization_summary.png
│   ├── metrics/
│   │   ├── model_comparison.csv
│   │   ├── final_model_benchmarks.csv
│   │   ├── hyperparameter_tuning.json
│   │   ├── feature_importance.csv
│   │   ├── optimization_summary.csv
│   │   └── regional_optimization.csv
│   └── predictions/
│       ├── test_set_predictions.csv
│       └── demonstration_cases.csv
└── docs/
    ├── Week4_Predictive_Modeling_and_Optimization.md
    └── Week4_Predictive_Modeling_and_Optimization.docx
```

---

## 🚀 Installation & Quickstart Execution

### 1. Clone & Set Up Environment:
```bash
git clone https://github.com/ajaym/week4-logistics-predictive-modeling.git
cd week4-logistics-predictive-modeling
pip install -r requirements.txt
```

### 2. Run the End-to-End ML & Optimization Pipeline:
```bash
python run_pipeline.py
```

### 3. Generate and Execute the Jupyter Notebook:
```bash
python build_and_execute_notebook.py
python -m jupyter nbconvert --to notebook --execute notebooks/week4_predictive_modeling.ipynb --output week4_predictive_modeling.ipynb
```

### 4. Compile the Word Academic Report:
```bash
python convert_docs_to_word.py
```

---

## 📈 Key Visualizations Generated

| Figure | Description | Output Path |
| :---: | :--- | :--- |
| **Fig 1** | Actual vs. Predicted Delivery Time Scatter Plot ($y=x$ Line) | `outputs/figures/01_actual_vs_predicted.png` |
| **Fig 2** | Residual Diagnostics vs. Fitted Delivery Time | `outputs/figures/02_residual_analysis.png` |
| **Fig 3** | Residual Density & Normal Q-Q Probability Plot | `outputs/figures/03_residual_distribution.png` |
| **Fig 4** | Model Benchmark Bar Charts (MAE, RMSE, $R^2$) | `outputs/figures/04_model_comparison.png` |
| **Fig 5** | 5-Fold Cross-Validation Stability Bar Chart | `outputs/figures/05_cross_validation_stability.png` |
| **Fig 6** | Top 10 Feature Importances Horizontal Bar Chart | `outputs/figures/06_feature_importance.png` |
| **Fig 7** | Hyperparameter Tuning Impact Comparison | `outputs/figures/07_hyperparameter_tuning_impact.png` |
| **Fig 8** | Cost vs. Delivery Lead Time Pareto Trade-Off Curve | `outputs/figures/08_optimization_cost_time_tradeoff.png` |
| **Fig 9** | Regional Logistics Cost: Baseline vs. Optimized Dispatch | `outputs/figures/09_regional_optimization_summary.png` |

---

## 💡 Key Business Recommendations

1. **Implement Dynamic Mode Selection**: Replace static rules with the LP optimization policy to realize **$15,289.87 in annual transportation cost savings (13.32%)**.
2. **Prioritize North and West Market Optimization**: Rebalance carrier contracts in the North and West zones to capture maximum regional efficiency gains (20.57% and 13.03% respectively).
3. **Dynamic SLA Quoting at Order Checkout**: Deploy the Ridge Regression API in the customer portal to provide precise delivery lead times rather than broad static windows.
4. **Volume-Tiered Carrier Negotiations**: Leverage the optimal freight volume allocations (`Standard Delivery`: 916 orders, `Express Air`: 334 orders) to negotiate bulk rate reductions with freight forwarders.

---

## 👨‍💻 Author & Project Credits

* **Author**: Ajay M
* **Role**: Senior Data Scientist, ML Engineer & Logistics Analytics Consultant
* **Academic Track**: Advanced Logistics Analytics & Machine Learning Internship
* **Submission Date**: August 2026
