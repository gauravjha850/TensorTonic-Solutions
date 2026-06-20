#include <cuda_runtime.h>
#include <math.h>

// Helper for block-level reduction of a single value using warp shuffle
__device__ float block_reduce_sum(float val) {
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    static __shared__ float s_mem[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        s_mem[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if the warp existed
    val = (threadIdx.x < blockDim.x / 32) ? s_mem[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // 1. One block handles exactly one row
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_inv_rms;
    int tid = threadIdx.x;

    // 2. Accumulate local sums of squares using a grid-stride loop across columns
    float local_sum_sq = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float val = input[row * N + col];
        local_sum_sq += val * val;
    }

    // Reduce across the block to find the total sum of squares for this row
    float total_sum_sq = block_reduce_sum(local_sum_sq);

    if (tid == 0) {
        // Calculate Mean Square and then Inverse RMS
        float mean_sq = total_sum_sq / N;
        s_inv_rms = rsqrtf(mean_sq + eps); // Fast inverse square root
    }
    __syncthreads();

    // 3. Final pass: Normalize and scale using gamma
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        output[idx] = input[idx] * s_inv_rms * gamma[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // Map one block to one row
    int blocks = M;
    
    // Select execution block thread count up to a maximum layout of 1024
    int threads = 256; 
    if (N < 256) {
        threads = ((N + 31) / 32) * 32; // Round up to nearest warp
        if (threads < 32) threads = 32;
    } else if (N > 256 && N <= 1024) {
        threads = ((N + 31) / 32) * 32;
    } else {
        threads = 1024; // Clamp block dimension limits
    }

    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}