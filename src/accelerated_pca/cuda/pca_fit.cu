#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUSOLVER_CHECK(err) \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSolver Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void centerData(float* X, const float* mean, int nRows, int nCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nRows && col < nCols) {
        X[row * nCols + col] -= mean[col];
    }
}

extern "C" void pcaCUDA(const float* h_X, int nRows, int nCols,
                        float* h_V, float* h_S) {

    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    float* d_X;
    CUDA_CHECK(cudaMalloc(&d_X, nRows * nCols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice));

    float* d_mean;
    CUDA_CHECK(cudaMalloc(&d_mean, nCols * sizeof(float)));
    float* d_ones;
    CUDA_CHECK(cudaMalloc(&d_ones, nRows * sizeof(float)));
    float* h_ones = new float[nRows];
    for(int i=0;i<nRows;i++) h_ones[i]=1.0f;
    CUDA_CHECK(cudaMemcpy(d_ones,h_ones,nRows*sizeof(float),cudaMemcpyHostToDevice));
    delete[] h_ones;

    const float alpha = 1.0f/nRows, beta=0.0f;
    CUBLAS_CHECK(cublasSgemv(cublasH,CUBLAS_OP_T,nRows,nCols,
                             &alpha,d_X,nRows,d_ones,1,&beta,d_mean,1));
    CUDA_CHECK(cudaFree(d_ones));

    dim3 block(16,16);
    dim3 grid((nCols+block.x-1)/block.x,(nRows+block.y-1)/block.y);
    centerData<<<grid,block>>>(d_X,d_mean,nRows,nCols);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_S, *d_VT, *d_work;
    int *devInfo;
    CUDA_CHECK(cudaMalloc(&d_S, nCols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_VT, nCols*nCols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devInfo,sizeof(int)));

    int lwork = 0;
    signed char jobu='N';
    signed char jobvt='A';
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH,nRows,nCols,&lwork));
    CUDA_CHECK(cudaMalloc(&d_work,lwork*sizeof(float)));

    CUSOLVER_CHECK(cusolverDnSgesvd(cusolverH,jobu,jobvt,nRows,nCols,
                                    d_X,nRows,
                                    d_S,
                                    nullptr, nRows,
                                    d_VT, nCols,
                                    d_work,lwork,
                                    nullptr, devInfo));

    CUDA_CHECK(cudaMemcpy(h_V,d_VT,nCols*nCols*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_S,d_S,nCols*sizeof(float),cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(devInfo));

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
}
