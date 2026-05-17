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

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit

  ---

# Setup

```bash
git clone <https://github.com/kundana-kolanuvada/telecom-customer-churn-prediction>

cd telecom_churn_project
```

---

Install Dependencies

```bash
pip install -r requirements.txt
```

---

Run Model Training

```bash
python train_model.py
```

This generates:
- `customer_churn_model.pkl`
- `encoders.pkl`

---
