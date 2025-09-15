import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
data = {'x':[3,5,8,9],
        'y':[6,7,5,10]}
df = pd.DataFrame(data)
print(df)

x = df-df.mean()
print(x)
cov_matrix = np.cov(x.T)
print("covariance_matrix", cov_matrix)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)
print("eigenvalues",pca.explained_variance_)
print("eigenvalues",pca.components_)
print("projection",x_pca)
