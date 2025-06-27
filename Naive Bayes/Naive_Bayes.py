import numpy as np

class NaiveBayes:
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.classes = np.unique(y)
    n_classes = len(self.classes)

    self.means = np.zeros((n_classes, n_features), dtype=np.float64)
    self.var = np.zeros((n_classes, n_features), dtype = np.float64)
    self.priors = np.zeros(n_classes, dtype=np.float64)

    for c in self.classes:
      X_c = X[y == c]
      self.means[c, :] = X_c.mean(axis=0)
      self.var[c, :] = X_c.var(axis=0)
      self.priors[c] = X_c.shape[0] / n_samples

  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def _predict(self, x):
    posteriors = []
    for idx, c in enumerate(self.classes):
      prior = np.log(self.priors[idx])
      class_conditional = self._pdf(idx, x)
      posterior = np.sum(np.log(class_conditional)) + prior
      posteriors.append(posterior)
    return self.classes[np.argmax(posteriors)]

  def _pdf(self, class_idx, x):
    mean = self.means[class_idx]
    var = self.var[class_idx]
    numerator = np.exp(-(x - mean) ** 2 / (2 * (var)))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator