import numpy as np
import pytest
from accelerated_pca.common import standardize

def test_standardize_basic():
    X = np.array([[1, 2], [3, 4]])
    X_std = standardize(X)
    assert np.allclose(np.mean(X_std, axis=0), 0)
    assert np.allclose(np.std(X_std, axis=0), 1)

def test_standardize_zero_variance():
    X = np.array([[1, 1], [1, 1]])
    X_std = standardize(X)
    assert np.allclose(X_std, np.zeros_like(X))
