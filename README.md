# Telecom Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn or not using customer service and subscription details.

## Project Overview

Customer churn prediction helps telecom companies identify customers who may discontinue their services. This project uses Machine Learning models to analyze customer behavior and predict churn probability.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Label Encoding
- SMOTE for handling class imbalance
- Model comparison using Cross Validation
- Random Forest classification
- Streamlit web application for prediction

---

## Tech Stack

| Category | Technology Used |
|---|---|
| Programming Language | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) |
| Data Analysis | ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) |
| Data Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=python&logoColor=white) |
| Machine Learning | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white) |
| Imbalanced Data Handling | ![SMOTE](https://img.shields.io/badge/SMOTE-FF6F00?logo=data&logoColor=white) |
| Additional ML Model | ![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?logo=xgboost&logoColor=white) |
| Deployment | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) |
| Notebook Environment | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) |
| Development Environment | ![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visual-studio-code&logoColor=white) |
| Model Serialization | ![Pickle](https://img.shields.io/badge/Pickle-4B8BBE?logo=python&logoColor=white) |

---

## Dataset

Dataset used: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


  ---

## Setup

```bash
git clone <https://github.com/kundana-kolanuvada/telecom-customer-churn-prediction>

cd telecom_churn_project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model Training

```bash
python train_model.py
```

This generates:
- `customer_churn_model.pkl`
- `encoders.pkl`

### Run Streamlit Application

```bash
streamlit run app.py
```
---
