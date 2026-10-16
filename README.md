# CodeAlpha - Sales Prediction using Python

## 📌 Overview
End-to-end Machine Learning pipeline to predict product sales based on advertising spend across TV, Radio, and Newspaper channels. Includes feature engineering, multiple regression models, hyperparameter tuning, and ensemble learning. Built as part of CodeAlpha Data Science Internship — Task 4.

## 🎯 Task
CodeAlpha Data Science Internship - Task 4

## 🛠 Techniques Used

- **Feature Engineering** — created 13 new features:
  - `total_spend` (sum of all ad channels)
  - Channel shares (`TV_share`, `Radio_share`, `Newspaper_share`)
  - Square root & log transforms of each channel
  - `tv_radio_interaction` (synergy feature)

- **Data Preprocessing** — median imputation + StandardScaler

- **8 Regression Models**:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - Polynomial Ridge (degree 2)
  - Decision Tree, Random Forest, Gradient Boosting

- **Hyperparameter Tuning** — RandomizedSearchCV with 5-Fold Cross-Validation

- **Ensemble Learning** — VotingRegressor + StackingRegressor with RidgeCV meta-learner

- **Evaluation** — R², RMSE, MAE, and MAPE

## 📊 Key Results

- **Best models**: Linear Regression, Ridge, Lasso (all ~0.92 R²)
- **Ensemble (Stacking)**: Best generalization
- **Top predictive features**:
  - `TV_share` (strongest impact on sales)
  - `Radio_share`
  - `Newspaper_share`
  - Original `TV` and `total_spend`

## 📈 Visualizations Generated

1. Feature correlation matrix (17×17 heatmap)
2. Sales vs Ad Spend scatter plots with regression lines
3. Sales distribution histogram with KDE
4. Model comparison chart (R² vs RMSE)
5. Actual vs Predicted + Residual plots
6. Top-15 Permutation feature importance
7. Learning curve (training vs validation R²)

## 🧰 Tech Stack

Python • Scikit-learn • Pandas • NumPy • Matplotlib • Seaborn • SciPy • Jupyter • Joblib

## 📂 Repository Structure
