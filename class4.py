import pandas as pd
import numpy as np
data={'age':[27,28,None,21,30,31,32,15,20,22],
     'height':[160,162,165,164,167,None,168,169,152,170],
     'wt':[56,57,58,59,60,65,81,61,62,65],
     'city':['kolkata','rampur','up','kanpur','bihar','kolkata','china','goa','up','mp'],
     'marks':[60,65,90,70,75,78,79,80,81,82]}
df=pd.DataFrame(data)
print(df)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ohe=OneHotEncoder(sparse_output=False)
ohe_array=ohe.fit_transform(df[['city']])
ohe_columns=ohe.get_feature_names_out(['city'])
df_ohe=pd.DataFrame(ohe_array,columns=ohe_columns)
print(df_ohe)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['le_city']=le.fit_transform(df['city'])
print(df)