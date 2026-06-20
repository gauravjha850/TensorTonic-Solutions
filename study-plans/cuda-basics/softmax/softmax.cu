#include <cuda_runtime.h>
#include <math.h>

// Cooperative block-level reduction helpers using warp shuffle
__device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float s_max;
    __shared__ float s_sum;

    // 1. Find the global maximum element across the array (Grid-stride loop)
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (input[i] > local_max) {
            local_max = input[i];
        }
    }
    
    // Reduce across the block to find the absolute max
    float max_val = block_reduce_max(local_max);
    if (threadIdx.x == 0) {
        s_max = max_val;
    }
    __syncthreads();

    // 2. Compute the sum of exponentials (with max subtracted for stability)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - s_max);
    }

    // Reduce across the block to find total sum
    float sum_val = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        s_sum = sum_val;
    }
    __syncthreads();

    // 3. Final pass: Compute the softmax probability and write to output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - s_max) / s_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // Launch a single block with 1024 threads to perform global reduction natively
    int threads = 1024;
    if (N < threads) {
        threads = ((N + 31) / 32) * 32; // Round up to nearest warp
    }
    
    softmax_kernel<<<1, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
