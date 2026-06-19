#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    // 1. One block handles one row
    int row = blockIdx.x;
    if (row >= M) return;

    // Shared memory allocations for block reduction
    __shared__ float s_sum;
    __shared__ float s_sum_sq;

    int tid = threadIdx.x;

    // Initialize shared memory
    if (tid == 0) {
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
    }
    __syncthreads();

    // 2. Accumulate local sums using a grid-stride loop across the row columns
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int col = tid; col < N; col += blockDim.x) {
        float val = input[row * N + col];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // 3. Atomically reduce within shared memory for simplicity and handling arbitrary block sizes
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    __syncthreads();

    // 4. Compute mean and inverse standard deviation on shared variables
    float mean = s_sum / N;
    float var = (s_sum_sq / N) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    // 5. Normalize elements and apply learnable affine transforms (gamma and beta)
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        output[idx] = (input[idx] - mean) * inv_std * gamma[col] + beta[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
