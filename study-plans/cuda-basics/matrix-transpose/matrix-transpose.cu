#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // 1. Calculate the 2D matrix coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of A
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of A
    
    // 2. Bounds-check elements within Matrix A
    if (i < M && j < N) {
        // Transpose mapping: B[j, i] = A[i, j]
        B[j * M + i] = A[i * N + j];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    // Launch the correct transpose kernel (Note: parameter 'B' replaces old 'C')
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
