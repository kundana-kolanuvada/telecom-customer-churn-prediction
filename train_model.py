import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split,cross_val_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
from imblearn.over_sampling import SMOTE

df = pd.read_csv("tel_churn.csv")

df = df.drop(columns=["customerID"])

df["TotalCharges"] = df["TotalCharges"].replace({" ": 0.0})

df["TotalCharges"] = df["TotalCharges"].astype(float)

object_columns = df.select_dtypes(include="object").columns
object_columns = object_columns.drop("Churn")

encoders = {}

for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

with open("encoders.pkl","wb") as f:
    pickle.dump(encoders,f)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}

for model_name, model in models.items():
    scores = cross_val_score(model,X_train_smote,y_train_smote,cv=5,scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} Accuracy: "f"{np.mean(scores):.2f}")

rfc = RandomForestClassifier(random_state=42)

rfc.fit(X_train_smote,y_train_smote)

y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n",accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n",classification_report(y_test,y_test_pred))

model_data = {"model": rfc,"feature_names":X.columns.to_list()}
with open("customer_churn_model.pkl","wb") as f:
    pickle.dump(model_data, f)
print("Model saved successfully.")