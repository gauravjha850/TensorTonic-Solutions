#include <cuda_runtime.h>
#include <math.h>

// Helper for block-level reduction of two synchronized values (sum and sum_sq)
__device__ void block_reduce_sum_pair(float &sum, float &sum_sq) {
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Shared memory allocated dynamically inside the kernel or at block level
    static __shared__ float s_mem_sum[32];
    static __shared__ float s_mem_sq[32];
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        s_mem_sum[wid] = sum;
        s_mem_sq[wid] = sum_sq;
    }
    __syncthreads();

    // Read from shared memory only if the warp existed
    sum = (threadIdx.x < blockDim.x / 32) ? s_mem_sum[lane] : 0.0f;
    sum_sq = (threadIdx.x < blockDim.x / 32) ? s_mem_sq[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }
}

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    // 1. One block handles one row
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_mean;
    __shared__ float s_inv_std;

    int tid = threadIdx.x;

    // 2. Accumulate local sums using a grid-stride loop across the row columns
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int col = tid; col < N; col += blockDim.x) {
        float val = input[row * N + col];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Reduce across the block to accumulate global sums for this row
    block_reduce_sum_pair(local_sum, local_sum_sq);

    if (tid == 0) {
        float mean = local_sum / N;
        // Population variance: E[X^2] - (E[X])^2
        float var = (local_sum_sq / N) - (mean * mean);
        if (var < 0.0f) var = 0.0f; // Guard against negative precision errors
        
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps); // Fast inverse square root
    }
    __syncthreads();

    // 3. Final pass: Normalize and apply learnable affine transforms
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        output[idx] = (input[idx] - s_mean) * s_inv_std * gamma[col] + beta[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    // Map one block to one row
    int blocks = M;
    
    // Choose execution width: clamp to 1024 or match layout column size
    int threads = 256; 
    if (N < 256) {
        threads = ((N + 31) / 32) * 32; // Round up to nearest warp boundary
        if (threads < 32) threads = 32;
    } else if (N > 256 && N <= 1024) {
        threads = ((N + 31) / 32) * 32;
    } else {
        threads = 1024; // Max threads per block block cap
    }

    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}