import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

data={
    'age':[27,28,29,None,30,31,32,None,20,22],
     'height':[160,162,165,None,167,None,168,169,None,170],
     'wt':[56,57,58,59,60,None,61,62,None,65],
     'city':['kolkata',np.nan,'up','bihar','kanpur',np.nan,'kolkata','china',np.nan,'bareilly'],
     'marks': [60,65,None,70,75,78,79,80,81,82]
     }
df1 = pd.DataFrame(data)
print(df1)

print("total missing_values in df1:\n", df1.isnull().sum())

df1_drop = df1.dropna()
print(df1_drop)

df1.to_csv('user1.csv', index=False)
df1_drop.to_csv('user1_drop.csv', index=False)

df1.loc[:, 'age'] = df1['age'].fillna(df1['age'].mean())
df1.loc[:, 'height'] = df1['height'].fillna(df1['height'].median())
df1.loc[:, 'wt'] = df1['wt'].ffill()
df1.loc[:, 'city'] = df1['city'].fillna(df1['city'].mode()[0])
df1.loc[:, 'marks'] = df1['marks'].bfill()
print(df1)

mm1 = MinMaxScaler()
std1 = StandardScaler()
df1[['mm_age', 'mm_height']] = mm1.fit_transform(df1[['age', 'height']])
df1[['std_wt']] = std1.fit_transform(df1[['wt']])
print(df1)

le = LabelEncoder()
df1['le_city'] = le.fit_transform(df1['city'])
print(df1)


# ---------------- Second Dataset ----------------
data2 = {
    'Patient_id': [101, 102, 103, 104, 105, 106, 107],
    'Age': [25, 30, 28, np.nan, 45, 50, 29],
    'wt': [58, 75, np.nan, 55, 85, 90, np.nan],
    'BP': [120, np.nan, 130, 110, 140, 150, np.nan],
    'Cholestrol': [200, 210, 220, 190, np.nan, 250, 230],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female']
}

df2 = pd.DataFrame(data2)
print(df2)

print("total missing_values in df2:\n", df2.isnull().sum())

df2_drop = df2.dropna()
print(df2_drop)

df2.to_csv('user2.csv', index=False)
df2_drop.to_csv('user2_drop.csv', index=False)

df2.loc[:, 'Age'] = df2['Age'].fillna(df2['Age'].mean())
df2.loc[:, 'wt'] = df2['wt'].fillna(df2['wt'].median())
df2.loc[:, 'BP'] = df2['BP'].fillna(df2['BP'].mean())
df2.loc[:, 'Cholestrol'] = df2['Cholestrol'].fillna(df2['Cholestrol'].median())
print(df2)

df2['le_Gender'] = le.fit_transform(df2['Gender'])
print(df2)

mm2 = MinMaxScaler()
std2 = StandardScaler()
df2[['mm_Age', 'mm_wt']] = mm2.fit_transform(df2[['Age', 'wt']])
df2[['std_Cholestrol']] = std2.fit_transform(df2[['Cholestrol']])
print(df2)
