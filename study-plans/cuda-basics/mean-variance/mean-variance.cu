#include <cuda_runtime.h>

// Kernel 1: Parallel tree reduction using a grid-stride loop to accumulate sum and sum of squares
__global__ void mean_var_reduce_kernel(const float* input, float* global_sum, float* global_sum_sq, int N) {
    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Grid-stride loop to comfortably process elements beyond the total grid size
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        float val = input[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // In-block shared memory parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 of each block atomically accumulates block results into global scratch targets
    if (tid == 0) {
        atomicAdd(global_sum, s_sum[0]);
        atomicAdd(global_sum_sq, s_sum_sq[0]);
    }
}

// Kernel 2: Single-threaded finalization to calculate biased population metrics
__global__ void mean_var_finalize_kernel(float* mean_out, float* var_out, const float* global_sum, const float* global_sum_sq, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = *global_sum / N;
        // Population Variance: E[X^2] - (E[X])^2
        float variance = (*global_sum_sq / N) - (mean * mean);
        
        // Guard against precision clipping errors producing tiny negative roots
        if (variance < 0.0f) variance = 0.0f;
        
        mean_out[0] = mean;
        var_out[0] = variance;
    }
}

// Host entry function
extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    // 1. Set up temporary variables to accumulate absolute intermediate values
    float *d_global_sum = nullptr, *d_global_sum_sq = nullptr;
    cudaMalloc(&d_global_sum, sizeof(float));
    cudaMalloc(&d_global_sum_sq, sizeof(float));
    
    cudaMemset(d_global_sum, 0, sizeof(float));
    cudaMemset(d_global_sum_sq, 0, sizeof(float));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Cap grid size to stay within execution limit thresholds
    
    // 2. Launch the global mapping collection pass
    mean_var_reduce_kernel<<<blocks, threads>>>(input, d_global_sum, d_global_sum_sq, N);
    
    // 3. Launch finalization pass to process final values safely
    mean_var_finalize_kernel<<<1, 1>>>(mean_out, var_out, d_global_sum, d_global_sum_sq, N);
    
    // 4. Synchronize GPU device timeline and clear temporary scratch pointers
    cudaDeviceSynchronize();
    cudaFree(d_global_sum);
    cudaFree(d_global_sum_sq);
}
