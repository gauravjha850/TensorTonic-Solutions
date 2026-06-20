#include <cuda_runtime.h>

// 1. CUDA Kernel to perform matrix-vector multiplication
__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    // Map thread to a unique output row index i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Guard to ensure we don't access out-of-bounds matrix rows
    if (i < M) {
        float sum = 0.0f;
        
        // 3. Serial loop over columns to accumulate the dot product
        for (int j = 0; j < N; ++j) {
            // Index the row-major matrix using: row * width + column
            sum += A[i * N + j] * x[j];
        }
        
        // Write the final calculated result to the output vector
        y[i] = sum;
    }
}

// Host entry function
extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    // Launch with a standard 256 threads per block as requested
    dim3 threads(256);
    
    // Calculate 1D grid size needed to cover all M rows
    dim3 blocks((M + 255) / 256);
    
    // Launch the GEMV execution kernel
    gemv_kernel<<<blocks, threads>>>(A, x, y, M, N);
    
    // Synchronize device execution
    cudaDeviceSynchronize();
}
