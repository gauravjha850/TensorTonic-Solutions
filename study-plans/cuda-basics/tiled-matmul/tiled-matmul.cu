#include <cuda_runtime.h>

#define TILE_DIM 16

// 1. CUDA Kernel to perform shared-memory tiled matrix multiplication
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for cooperative tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Map thread indices to global output matrix coordinates
    int row = blockIdx.y * TILE_DIM + ty; // Row index for C and A
    int col = blockIdx.x * TILE_DIM + tx; // Column index for C and B

    // Local register accumulator for the final dot product element
    float value = 0.0f;

    // Loop over the shared dimension K in tile-sized steps
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        
        // Phase 1: Cooperatively load tiles from global memory into shared memory
        // Guard boundaries for Matrix A: (row < M) and (global K index < K)
        int k_global_A = t * TILE_DIM + tx;
        if (row < M && k_global_A < K) {
            As[ty][tx] = A[row * K + k_global_A];
        } else {
            As[ty][tx] = 0.0f; // Requirements check: Write 0 for out-of-range rows/columns
        }

        // Guard boundaries for Matrix B: (global K index < K) and (col < N)
        int k_global_B = t * TILE_DIM + ty;
        if (k_global_B < K && col < N) {
            Bs[ty][tx] = B[k_global_B * N + col];
        } else {
            Bs[ty][tx] = 0.0f; // Requirements check: Write 0 for out-of-range rows/columns
        }

        // Synchronize 1: Wait for all threads to finish filling the shared memory tiles
        __syncthreads();

        // Phase 2: Compute partial products from the loaded tile arrays
        for (int k = 0; k < TILE_DIM; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        // Synchronize 2: Ensure computation finishes before loading the next tile phase
        __syncthreads();
    }

    // Phase 3: Write the finalized dot product back to global memory C
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Host entry function
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // Configure a 16x16 execution thread block (TILE_DIM = 16)
    dim3 threads(TILE_DIM, TILE_DIM);

    // Calculate grid layout sizes to fully encompass output dimensions (M, N)
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the tiled matrix multiplication kernel
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);

    // Synchronize execution stream
    cudaDeviceSynchronize();
}