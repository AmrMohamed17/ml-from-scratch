from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from LR import LR

def accuracy(y_true, y_pred):
  return np.mean(y_true == y_pred)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LR(lr=0.0001, epochs=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))