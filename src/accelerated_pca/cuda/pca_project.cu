#include <cuda_runtime.h>
#include <iostream>

__global__ void projectData(const float* X, const float* mean,
                            const float* components, float* X_proj,
                            int nRows, int nCols, int nComponents)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < nRows && col < nComponents)
    {
        float sum = 0.0f;
        for (int k = 0; k < nCols; ++k)
            sum += (X[row * nCols + k] - mean[k]) * components[col * nCols + k];

        X_proj[row * nComponents + col] = sum;
    }
}

extern "C" void projectCUDA(const float* h_X, const float* h_mean,
                            const float* h_components, float* h_X_proj,
                            int nRows, int nCols, int nComponents)
{
    float *d_X, *d_mean, *d_components, *d_X_proj;
    cudaMalloc(&d_X, nRows*nCols*sizeof(float));
    cudaMalloc(&d_mean, nCols*sizeof(float));
    cudaMalloc(&d_components, nComponents*nCols*sizeof(float));
    cudaMalloc(&d_X_proj, nRows*nComponents*sizeof(float));

    cudaMemcpy(d_X, h_X, nRows*nCols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, nCols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_components, h_components, nComponents*nCols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((nComponents + block.x - 1)/block.x, (nRows + block.y - 1)/block.y);

    projectData<<<grid, block>>>(d_X, d_mean, d_components, d_X_proj, nRows, nCols, nComponents);
    cudaDeviceSynchronize();

    cudaMemcpy(h_X_proj, d_X_proj, nRows*nComponents*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_mean);
    cudaFree(d_components);
    cudaFree(d_X_proj);
}
