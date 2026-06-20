#include <cuda_runtime.h>

// 1. CUDA Kernel to perform element-wise addition
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    // Calculate the unique global thread ID across the entire grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Guard to ensure we don't access memory out of the vector bounds
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 2. Host function to configure grid and launch the kernel
extern "C" void solve(const float *A, const float *B, float *C, int N) {
    // Define block size (number of threads per block)
    int threads = 256;
    
    // Calculate grid size (number of blocks needed to cover N elements)
    int blocks = (N + threads - 1) / threads;
    
    // Launch the kernel on the GPU
    vector_add<<<blocks, threads>>>(A, B, C, N);
    
    // Synchronize device execution to ensure accuracy before moving forward
    cudaDeviceSynchronize();
}