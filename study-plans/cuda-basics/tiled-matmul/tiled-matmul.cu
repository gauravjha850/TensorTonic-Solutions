#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Map thread indices to global matrix coordinates
    int row = blockIdx.y * TILE_DIM + ty; // Row index for C and A
    int col = blockIdx.x * TILE_DIM + tx; // Column index for C and B

    // Running register accumulator for the output element C[row, col]
    float value = 0.0f;

    // Loop over the shared dimension K in tile-sized steps
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        
        // 1. Cooperatively load one element of the A tile into shared memory
        int k_idx_A = t * TILE_DIM + tx;
        if (row < M && k_idx_A < K) {
            As[ty][tx] = A[row * K + k_idx_A];
        } else {
            As[ty][tx] = 0.0f; // Padding out-of-range positions
        }

        // 2. Cooperatively load one element of the B tile into shared memory
        int k_idx_B = t * TILE_DIM + ty;
        if (k_idx_B < K && col < N) {
            Bs[ty][tx] = B[k_idx_B * N + col];
        } else {
            Bs[ty][tx] = 0.0f; // Padding out-of-range positions
        }

        // Synchronize to make sure both tiles are completely populated
        __syncthreads();

        // Accumulate the products of the cached sub-vectors from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to protect shared memory before loading the next tile set
        __syncthreads();
    }

    // Write the final accumulated value to matrix C if within valid bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    
    cudaDeviceSynchronize();
}