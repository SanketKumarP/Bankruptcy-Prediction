
# 🏦 Bankruptcy Prediction using Machine Learning

A comprehensive machine learning pipeline developed to predict corporate bankruptcy using financial ratios and advanced classification models. This project was completed as part of the Statistical Learning (MATH-569) course in Spring 2024.

## 📌 Project Overview

Corporate bankruptcy prediction is a critical task in the financial domain, where early identification of at-risk firms can lead to significant economic safeguards. In this project, we implement and compare multiple models, address class imbalance, and tackle multicollinearity to build a robust predictive framework.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://doi.org/10.24432/C5004D)
- **Size**: 6819 companies, 95 financial indicators
- **Period**: 10 years
- **Target**: `Bankrupt?` (1 = Yes, 0 = No)

---

## 🔁 Workflow

### 1. Data Preprocessing
- Removed uninformative columns
- Applied **Z-standardization**
- Addressed **class imbalance** with **SMOTE**

### 2. Exploratory Data Analysis
- Feature correlation
- Detection of multicollinearity via **VIF**

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)** used to reduce feature space and mitigate collinearity

### 4. Feature Selection
- Applied **Recursive Feature Elimination (RFE)** on various models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - AdaBoost
  - Gradient Boosting

### 5. Model Development
- Logistic Regression with PCA
- XGBoost with full and selected features
- Ensemble models (Random Forest, Gradient Boosting)

### 6. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Cross-validation (5-fold)

---

## 📈 Key Results

| Model                    | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| Benchmark XGBoost       | 0.828    | 0.286     | 0.078  | 0.123    |
| Logistic Regression + PCA | 0.967  | —         | —      | —        |
| XGBoost (Full features) | 0.9818   | —         | —      | —        |

> XGBoost and Random Forest with RFE-selected features achieved the best balance of performance and consistency.

---

## 🔍 Interpretability vs. Accuracy

- **Linear models** (e.g., Logistic Regression) provide interpretability but may underperform.
- **Tree-based models** (e.g., XGBoost, RF) excel in accuracy but act as black boxes.
- **Model choice should align with business context and need for explainability.**

---

## 🔮 Future Work

- Explore **L1/L2 Regularization** for linear models
- Introduce **SVMs** and **Neural Networks**
- Apply **cost-sensitive learning**
- Engineer **temporal and interaction features**

---

## 📚 References

1. [Taiwanese Bankruptcy Dataset – UCI](https://doi.org/10.24432/C5004D)  
2. [Musa et al. (2022)](https://www.mdpi.com/1911-8074/15/1/35)  
3. [SMOTE: Chawla et al. (2002)](https://jair.org/index.php/jair/article/view/10302)  
4. [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
5. [Scikit-learn RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
