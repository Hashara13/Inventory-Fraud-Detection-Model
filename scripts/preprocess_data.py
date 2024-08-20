import pandas as pd
import os

physical_data=pd.read.data(os.path.join('data','raw','physical_inventory.csv'))
erp_data=pd.read.data(os.path.join('data','raw','erp_inventory.csv'))