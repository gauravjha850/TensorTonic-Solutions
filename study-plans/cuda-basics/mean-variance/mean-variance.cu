#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    if (idx < N) {
        float val = input[idx];
        local_sum = val;
        local_sum_sq = val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate block totals into global memory
    if (tid == 0) {
        atomicAdd(mean_out, s_sum[0]);
        atomicAdd(var_out, s_sum_sq[0]);
    }
}

__global__ void finalize_kernel(float* mean_out, float* var_out, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = mean_out[0];
        float sum_sq = var_out[0];
        
        float mu = sum / N;
        // Var(X) = E[X^2] - (E[X])^2
        float variance = (sum_sq / N) - (mu * mu);
        
        mean_out[0] = mu;
        var_out[0] = variance;
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));
    
    // Step 1: Accumulate global sum and sum of squares
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    
    // Step 2: Compute final mean and variance from the accumulated aggregates
    finalize_kernel<<<1, 1>>>(mean_out, var_out, N);
    
    cudaDeviceSynchronize();
}
