import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import joblib
import os

data_path = '../data/processed/merged_inventory.csv'
data = pd.read_csv(data_path)

np.random.seed(42)
data['Fraud_Flag'] = np.random.randint(0, 2, size=len(data))

features = ['Discrepancy', 'Days_Since_Adjustment']
target = 'Fraud_Flag'

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
