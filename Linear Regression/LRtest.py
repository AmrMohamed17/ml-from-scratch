import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LR import LinearRegression
import matplotlib.pyplot as plt

x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

regression = LinearRegression(lr=0.01, num_iter=1000)
regression.fit(X_train, y_train)
predictions = regression.predict(X_test)
# Calculate Mean Squared Error
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

# Optional: Plotting the results
y_pred = regression.predict(x)
fig = plt.figure(figsize=(9, 6))
m1 = plt.scatter(X_train, y_train, cmap=(0.9), s=10)
m2 = plt.scatter(X_test, y_test, cmap=(0.5), s=10)
plt.plot(x, y_pred, color='black', linewidth=2, label='Prediction')
plt.show()

