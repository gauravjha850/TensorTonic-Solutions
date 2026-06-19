#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    // 1. Calculate the 2D matrix coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index (0 to N-1)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (0 to M-1)
    
    // 2. Bounds check to make sure threads stay within the M x N matrix dimensions
    if (i < M && j < N) {
        // C[i, j] = a[i] * b[j]
        C[i * N + j] = a[i] * b[j];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    // Define a 16x16 thread block structure
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    // Launch the outer product kernel
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
