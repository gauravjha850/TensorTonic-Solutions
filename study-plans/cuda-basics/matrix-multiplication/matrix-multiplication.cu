#include <cuda_runtime.h>

// 1. CUDA Kernel to perform matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Determine the row (i) and column (j) indices for the current thread
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index in C and B
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index in C and A

    // 2. Guard against out-of-bounds execution
    if (i < M && j < N) {
        float sum = 0.0f;
        
        // 3. Loop over the shared dimension K to calculate the dot product
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        
        // Write the calculated result to the row-major output matrix
        C[i * N + j] = sum;
    }
}

// Host entry function
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // Configure a 16x16 execution thread block as required
    dim3 threads(16, 16);
    
    // Calculate block count to adequately cover the entire output matrix dimensions
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    // Launch the core matrix multiplication kernel
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    
    // Wait for the GPU to finish execution before exiting
    cudaDeviceSynchronize();
}
