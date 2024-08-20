import pandas as pd
import os

physical_data = pd.read_csv(os.path.join('data', 'raw', 'physical_inventory.csv'))
erp_data = pd.read_csv(os.path.join('data', 'raw', 'erp_inventory.csv'))

physical_data['Date'] = pd.to_datetime(physical_data['Date'], errors='coerce')
erp_data['Date'] = pd.to_datetime(erp_data['Date'], errors='coerce')

merged_data = pd.merge(physical_data, erp_data, on=['Date', 'Item_Code', 'Warehouse_ID'])
merged_data['Discrepancy'] = merged_data['Physical_Quantity'] - merged_data['ERP_Quantity']
merged_data['Significant_Discrepancy'] = merged_data['Discrepancy'].abs() > 10
merged_data['Days_Since_Adjustment'] = (merged_data['Date'] - pd.to_datetime(merged_data['Last_Adjustment'])).dt.days
merged_data.to_csv(os.path.join('data', 'processed', 'merged_inventory.csv'), index=False)
