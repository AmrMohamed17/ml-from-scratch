import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class LR:
  def __init__(self, lr=0.001, epochs=1000):
    self.lr = lr
    self.epochs = epochs
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    self.weights = np.zeros(X.shape[1])
    self.bias = 0

    for _ in range(self.epochs):
      lr_model = np.dot(X, self.weights) + self.bias
      y_hat = sigmoid(lr_model)

      dw = (np.dot(X.T, (y_hat - y))) / X.shape[0]
      db = (np.sum(y_hat - y)) / X.shape[0]

      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    lr_model = np.dot(X, self.weights) + self.bias
    y_hat = sigmoid(lr_model)
    y_pred = [1 if i >= 0.5 else 0 for i in y_hat]
    return np.array(y_pred)

