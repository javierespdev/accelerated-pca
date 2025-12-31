import numpy as np
from typing import Optional

class PCA_CPU:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx]
        eigen_values = eigen_values[idx]  # <-- guardar orden

        k = self.n_components or eigen_vectors.shape[1]
        self.components_ = eigen_vectors[:, :k].T
        self.singular_values_ = np.sqrt(eigen_values[:k] * (X.shape[0]-1))

        return self


    def transform(self, X: np.ndarray):
        if self.components_ is None:
            raise RuntimeError("PCA not fitted yet")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)
