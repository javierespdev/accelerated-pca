#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cmath>

extern "C" {

__global__ void center_kernel(float* X, const float* mean, int nRows, int nCols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows * nCols) {
        X[idx] -= mean[idx % nCols];
    }
}

__global__ void mean_kernel(const float* X, float* mean, int nRows, int nCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < nCols) {
        float sum = 0.0f;
        for (int i = 0; i < nRows; ++i) sum += X[i * nCols + col];
        mean[col] = sum / nRows;
    }
}

int pcaCUDA(float* h_X, int nRows, int nCols, int nComponents, float* h_mean, float* h_components, float* h_S, float* elapsed_ms) {
    cublasHandle_t cublas;
    cusolverDnHandle_t solver;
    cublasCreate(&cublas);
    cusolverDnCreate(&solver);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    float *d_X, *d_mean, *d_Cov, *d_W, *d_work;
    int *devInfo, lwork;
    cudaMalloc(&d_X, nRows * nCols * sizeof(float));
    cudaMalloc(&d_mean, nCols * sizeof(float));
    cudaMalloc(&d_Cov, nCols * nCols * sizeof(float));
    cudaMalloc(&d_W, nCols * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    cudaMemcpy(d_X, h_X, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Center data
    mean_kernel<<<(nCols + 255) / 256, 256>>>(d_X, d_mean, nRows, nCols);
    center_kernel<<<(nRows * nCols + 255) / 256, 256>>>(d_X, d_mean, nRows, nCols);

    // 2. Covariance matrix C = (X^T * X) / (nRows - 1)
    float alpha = 1.0f / (nRows - 1.0f), beta = 0.0f;
    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, nCols, nCols, nRows, &alpha, d_X, nCols, d_X, nCols, &beta, d_Cov, nCols);

    // 3. Eigendecomposition (Eigenvectors)
    cusolverDnSsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nCols, d_Cov, nCols, d_W, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));
    cusolverDnSsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nCols, d_Cov, nCols, d_W, d_work, lwork, devInfo);

    cudaMemcpy(h_mean, d_mean, nCols * sizeof(float), cudaMemcpyDeviceToHost);
    float *t_W = new float[nCols];
    float *t_V = new float[nCols * nCols];
    cudaMemcpy(t_W, d_W, nCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(t_V, d_Cov, nCols * nCols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nComponents; i++) {
        h_S[i] = sqrtf(fmaxf(0.0f, t_W[nCols - 1 - i] * (nRows - 1)));
        for (int j = 0; j < nCols; j++) {
            h_components[i * nCols + j] = t_V[(nCols - 1 - i) * nCols + j];
        }
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_ms, start, stop);

    delete[] t_W; delete[] t_V;
    cudaFree(d_X); cudaFree(d_mean); cudaFree(d_Cov); cudaFree(d_W); cudaFree(d_work); cudaFree(devInfo);
    cublasDestroy(cublas); cusolverDnDestroy(solver);
    return 0;
}

int projectCUDA(float* h_X, int nRows, int nCols, int nComponents, float* h_mean, float* h_components, float* h_X_proj, float* elapsed_ms) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    float *d_X, *d_mean, *d_V, *d_Out;
    cudaMalloc(&d_X, nRows * nCols * sizeof(float));
    cudaMalloc(&d_mean, nCols * sizeof(float));
    cudaMalloc(&d_V, nComponents * nCols * sizeof(float));
    cudaMalloc(&d_Out, nRows * nComponents * sizeof(float));

    cudaMemcpy(d_X, h_X, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, nCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_components, nComponents * nCols * sizeof(float), cudaMemcpyHostToDevice);

    center_kernel<<<(nRows * nCols + 255) / 256, 256>>>(d_X, d_mean, nRows, nCols);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                nComponents, nRows, nCols, 
                &alpha, 
                d_V, nComponents, 
                d_X, nCols, 
                &beta, 
                d_Out, nComponents);

    cudaMemcpy(h_X_proj, d_Out, nRows * nComponents * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_ms, start, stop);

    cudaFree(d_X); cudaFree(d_mean); cudaFree(d_V); cudaFree(d_Out);
    cublasDestroy(handle);
    return 0;
}
}
