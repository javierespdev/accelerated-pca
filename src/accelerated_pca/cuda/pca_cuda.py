import ctypes
import numpy as np
import os
from typing import Optional

class PCA_CUDA:
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

        base_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(base_dir, "libpca_cuda.so")
        self.lib = ctypes.CDLL(so_path)

        self.lib.pcaCUDA.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # X
            ctypes.c_int,                    # nRows
            ctypes.c_int,                    # nCols
            ctypes.POINTER(ctypes.c_float),  # V
            ctypes.POINTER(ctypes.c_float)   # S
        ]
        self.lib.pcaCUDA.restype = None

        self.lib.projectCUDA.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # X
            ctypes.POINTER(ctypes.c_float),  # mean
            ctypes.POINTER(ctypes.c_float),  # components
            ctypes.POINTER(ctypes.c_float),  # X_proj
            ctypes.c_int,                    # nRows
            ctypes.c_int,                    # nCols
            ctypes.c_int                     # nComponents
        ]
        self.lib.projectCUDA.restype = None

    def fit(self, X: np.ndarray):
        X = np.ascontiguousarray(X, dtype=np.float32)
        nRows, nCols = X.shape
        X_flat = X.ravel()

        V = np.zeros(nCols * nCols, dtype=np.float32)
        S = np.zeros(nCols, dtype=np.float32)

        self.lib.pcaCUDA(
            X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nRows,
            nCols,
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            S.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        V = V.reshape(nCols, nCols)

        if self.n_components is None:
            self.components_ = V.T
        else:
            self.components_ = V.T[:self.n_components]

        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.singular_values_ = S

        return self

    def transform(self, X: np.ndarray):
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA not fitted yet")

        X = np.ascontiguousarray(X, dtype=np.float32)
        nRows, nCols = X.shape
        nComponents = self.components_.shape[0]

        X_proj = np.zeros((nRows, nComponents), dtype=np.float32)

        self.lib.projectCUDA(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.mean_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.components_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            X_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nRows,
            nCols,
            nComponents
        )

        return X_proj

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)
