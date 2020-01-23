import pandas as pd 
import json
import random

df=pd.read_csv('data/train.csv')
df_ = df.copy()
for f in df_.columns:
    if df_[f].dtype=='object': 
        del df_[f]
del df_['Id']    
del df_['SalePrice']  
# Fill in the missing data
df_.fillna(0, inplace=True)
#print (len(df_.columns))
idx = random.randint(0, len(df_))
sample_df = df_.iloc[idx]
sample_json = sample_df.to_json()
print (sample_json)

