import numpy as np
import pytest
from accelerated_pca import PCA_CPU

@pytest.mark.parametrize("shape", [(10, 5), (100, 50), (1000, 200)])
@pytest.mark.parametrize("n_components", [1, 3, None])
def test_cpu_fit_transform_equivalence(shape, n_components):
    np.random.seed(42)
    X = np.random.rand(*shape).astype(np.float32)
    
    pca = PCA_CPU(n_components=n_components)
    X_transformed1 = pca.fit_transform(X)
    X_transformed2 = pca.fit(X).transform(X)
    
    assert np.allclose(X_transformed1, X_transformed2, atol=1e-5)
