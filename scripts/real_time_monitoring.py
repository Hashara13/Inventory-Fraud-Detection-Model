import joblib
import pandas as pd

model = joblib.load('models/fraud_detection_model.pkl')

def detect_fraud(physical_qty, erp_qty, days_since_adj):
    discrepancy = physical_qty - erp_qty
    features = [[discrepancy, days_since_adj]]
    return model.predict(features)[0]

real_time_data = pd.DataFrame({
    'Item_Code': ['TEX001', 'TEX002', 'TEX003'],
    'Physical_Quantity': [150, 200, 105],
    'ERP_Quantity': [140, 210, 100],
    'Days_Since_Adjustment': [2, 4, 3]
})

real_time_data['Fraud_Detected'] = real_time_data.apply(
    lambda x: detect_fraud(x['Physical_Quantity'], x['ERP_Quantity'], x['Days_Since_Adjustment']), axis=1
)

for index, row in real_time_data.iterrows():
    if row['Fraud_Detected']:
        print(f"Fraud detected for Item {row['Item_Code']}!")
