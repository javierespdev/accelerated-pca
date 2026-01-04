import ctypes
import numpy as np
import os

class PCA_CUDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        
        dll_path = os.path.join(os.path.dirname(__file__), "libpca_cuda.so")
        self.lib = ctypes.CDLL(dll_path)
        
        self.lib.pcaCUDA.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.projectCUDA.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
        ]

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        r, c = X.shape
        self.mean_ = np.zeros(c, dtype=np.float32)
        self.components_ = np.zeros((self.n_components, c), dtype=np.float32)
        S = np.zeros(self.n_components, dtype=np.float32)
        ms = np.zeros(1, dtype=np.float32)

        self.lib.pcaCUDA(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c, self.n_components,
                         self.mean_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         self.components_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         S.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         ms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self

    def transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        r, c = X.shape
        out = np.zeros((r, self.n_components), dtype=np.float32)
        ms = np.zeros(1, dtype=np.float32)

        self.lib.projectCUDA(X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), r, c, self.n_components,
                            self.mean_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            self.components_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            ms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
