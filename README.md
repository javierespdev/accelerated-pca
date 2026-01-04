# Accelerated PCA

**Accelerated PCA** is a Python package implementing **Principal Component Analysis (PCA)** optimized for both CPU and GPU using **CUDA**. It allows fast dimensionality reduction and easy comparison between CPU and GPU performance.


<p align="center">
  <img src="assets/animation/pca_projection.gif" width="600">
</p>

## Overview

Principal Component Analysis (PCA) is a technique used to **reduce the dimensionality** of a dataset while preserving as much variance as possible. This package provides:

- **PCA_CPU**: Fast implementation for CPUs.  
- **PCA_CUDA**: GPU-accelerated implementation using CUDA.  
- Easy benchmarking and comparison between CPU and GPU.

## Usage Example

```py
import numpy as np
from accelerated_pca import PCA_CPU, PCA_CUDA

X = np.random.rand(100, 50)

# CPU
pca_cpu = PCA_CPU(n_components=10)
X_cpu = pca_cpu.fit_transform(X)

# GPU
pca_gpu = PCA_CUDA(n_components=10)
X_gpu = pca_gpu.fit_transform(X)
```

## Steps of PCA

1. **Standardization**  
   - Normalize each feature to have zero mean and unit variance.  

        ![Standardization Formula](assets/equations/standardization.svg)

2. **Compute Covariance Matrix and Eigenvectors**  
   - Captures the relationships between features.  

        ![Covariance Matrix Formula](assets/equations/covariance_matrix.svg)

   - Solve:

        ![Covariance Eigen Formula](assets/equations/covariance_eigen_equation.svg)

   - Eigenvectors (v) indicate directions of maximum variance, eigenvalues (λ) their magnitude.

3. **Select Principal Components**  
   - Choose the first (k) eigenvectors depending on the desired explained variance.  

4. **Project Data onto Components**  
   - Transform original data:  

        ![PCA Projection](assets/equations/pca_projection.svg)

   - (V_k) are the selected top (k) eigenvectors.

## Benchmark: CPU PCA vs CUDA PCA vs Sklearn PCA

**GPU-accelerated PCA implementation** is 3.5–4× faster than sklearn's CPU PCA on large datasets.

![Benchmark](assets/benchmark/benchmark.png)

## Installation

To quickly set up the environment and run the benchmarks or tests, use the following commands:

```bash
git clone https://github.com/javierespdev/accelerated-pca.git
uv venv
source .venv/bin/activate
uv sync
mise run
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
