import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


# Dataset

data = {
    'Patient_ID':[101,102,103,104,105,106,107],
    'Age':[25,30,28,np.nan,45,50,29],
    'Weight':[58,75,np.nan,55,85,90,np.nan],
    'BloodPressure':[120,np.nan,130,110,140,150,np.nan],
    'Cholesterol':[200,210,220,190,np.nan,250,230],
    'Gender':['Male','Female','Male','Female','Male','Male','Female']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")

# Q1. Handling Missing Values


# 1. Count missing values
print("Missing Values Count:\n", df.isnull().sum(), "\n")



# 2. Replace missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)                # mean
df['Weight'].fillna(df['Weight'].median(), inplace=True)        # median
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].median(), inplace=True)

print("After Handling Missing Values:\n", df, "\n")






# Q2. Encoding
# Label Encoding Gender
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])

print("After Encoding:\n", df[['Gender','Gender_Encoded']], "\n")
print("Encoding Classes:", le.classes_)  





# Q3. Feature Scaling


# Min-Max Normalization (0-1 scale) for Age and Weight
scaler_minmax = MinMaxScaler()
df[['Age_MinMax','Weight_MinMax']] = scaler_minmax.fit_transform(df[['Age','Weight']])




# Standardization (Z-score) for Cholesterol
scaler_standard = StandardScaler()
df['Cholesterol_Z'] = scaler_standard.fit_transform(df[['Cholesterol']])
print("After Scaling:\n", df, "\n")



# Verify Cholesterol Z-score
print("Cholesterol_Z Mean:", round(df['Cholesterol_Z'].mean(), 2))
print("Cholesterol_Z Std Dev:", round(df['Cholesterol_Z'].std(), 2))
