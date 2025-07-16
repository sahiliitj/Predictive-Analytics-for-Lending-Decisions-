# Applied Analytics for Predictive Lending

> **Course:** MSL71440 – Analytics Lab (Python), Autumn 2024  
> **Institution:** Indian Institute of Technology, Jodhpur  
> **Authors:** Aman Kanshotia (M21MA201), Sahil (M21MA210), Abhas Malguri (M23MA1001)  
> **Program:** M.Sc – M.Tech (Data and Computational Sciences), Department of Mathematics

---

## 📌 Project Overview

This project aims to develop robust analytical models for **automated loan approval classification** and **credit risk score prediction** using a comprehensive dataset simulating real-world borrower profiles. The pipeline integrates rigorous **exploratory data analysis (EDA)**, **feature engineering**, and **supervised machine learning techniques** to support **data-driven lending decisions**.

---

## 📂 Dataset Description

The dataset comprises 20,000 loan applicants and includes the following components:

- **Demographics**: Age, Gender, Marital Status, Dependents, Education
- **Financials**: Income, Savings, Liabilities, Assets, Monthly Expenses
- **Credit History**: Credit Score, Defaults, Inquiries, Payment Behavior
- **Loan Details**: Amount, Purpose, Duration, Interest Rate
- **Outcome Variables**:
  - `Loan Approved` (Binary Classification)
  - `Risk Score` (Regression)

---

## 📊 Exploratory Data Analysis

Comprehensive visual and statistical analysis was conducted to understand patterns in:

- ✅ **Loan approval distribution** across demographics and education
- 💰 **Income and loan amount** distributions by education and employment
- 📉 **Credit score trends** segmented by employment status
- 🏡 **Impact of home ownership** and **loan purpose** on approval decisions

> Visualizations include bar plots, KDEs, violin plots, box plots, and AUC-ROC curves.

---

## 🔄 Data Preprocessing Pipeline

Utilizing `NumPy`, `Pandas`, and `scikit-learn`, we designed a reusable pipeline:

- Missing value handling and column separation (numerical vs categorical)
- Standardization via `StandardScaler`
- One-hot encoding with `drop='first'`
- Train-Test split (80-20) with reproducibility
- Encapsulated in a `Pipeline` object for transformation and model integration

---

## 🧠 Machine Learning Models

### 🎯 Classification: Loan Approval

| Model               | Accuracy | Precision | Recall  | F1-Score | Cross-Val Accuracy |
|--------------------|----------|-----------|---------|----------|---------------------|
| Logistic Regression| **99.98%** | 99.90%    | 100.00% | **99.95%** | 99.96%              |
| Random Forest      | 99.10%   | 98.52%    | 97.94% | 98.22%   | 98.82%              |
| Decision Tree      | 99.00%   | 98.13%    | 97.94% | 98.03%   | 98.94%              |
| KNN                | 95.98%   | 97.98%    | 85.94% | 91.57%   | 95.74%              |
| Naive Bayes        | 95.35%   | 87.67%    | 95.08% | 91.23%   | 95.45%              |

📌 **Best Classifier**: Logistic Regression achieved perfect recall, indicating it successfully identified all loan approvals with minimal false negatives.

### 📈 Regression: Risk Score Prediction

| Model                 | Validation R² | Validation MAE | Validation MSE |
|----------------------|----------------|----------------|----------------|
| XGBoost Regressor     | **0.9743**     | 0.0143         | **0.0046**     |
| Random Forest Regressor| 0.9715        | 0.0138         | 0.0051         |
| SVR                   | 0.9405        | 0.0760         | 0.0106         |
| Decision Tree         | 0.9476        | 0.0093         | 0.0094         |
| Linear Regression     | 0.8736        | 0.1201         | 0.0226         |

📌 **Best Regressor**: XGBoost Regressor demonstrated superior performance in predicting risk scores, balancing generalization and accuracy.

---

## 🧪 Technical Stack

- **Language**: Python 3.x  
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`
- **Development**: Jupyter Notebook, Python Scripts
- **Pipeline**: `Pipeline` + `ColumnTransformer` for modular preprocessing

---


---

## 📈 Key Insights

- Loan approvals were significantly influenced by **employment status**, **education**, and **loan purpose**
- The **Logistic Regression model** achieved near-perfect performance with minimal complexity
- **XGBoost** demonstrated exceptional accuracy for **risk score estimation**
- The pipeline architecture supports reproducibility and scalable experimentation

---

## 📚 References

- Lecture materials and dataset provided by the **MSL71440 course instructors**
- Scikit-learn and XGBoost official documentation
- Research papers on credit risk modeling and scorecards

---

## 🧾 License

This repository is for academic and research purposes only. Usage beyond educational contexts must receive prior approval from the contributors.

---

## 🤝 Contact

For academic collaboration or questions, please contact:

- **Aman Kanshotia** – `m21ma201@alumni.iitj.ac.in`
- **Sahil** – `m21ma210@alumni.iitj.ac.in`
- **Abhas Malguri** – `m23ma1001@alumni.iitj.ac.in`

---



