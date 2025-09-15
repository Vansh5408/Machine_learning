import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data
x = [[40,20],[50,50],[60,90],[10,25],[70,70],[60,10],[25,80]]
y = ["red","blue","blue","red","blue","red","blue"]

# Test point
q = [[20,35]]

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)

# Prediction
prediction = knn.predict(q)

# Plot training points
for (x1, y1), label in zip(x, y):
    color = "red" if label == "red" else "blue"
    plt.scatter(x1, y1, c=color)

# Plot query point
qx, qy = q[0]
plt.scatter(qx, qy, color="orange", marker="*", s=200)

# Labels and title
plt.title(f"Test data prediction = {prediction[0]}")
plt.xlabel("brightness")
plt.ylabel("saturation")
plt.grid(True)
plt.show()
