#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    // 1. One block handles exactly one row
    int row = blockIdx.x;
    if (row >= M) return;

    // Shared memory allocation for block-wide sum of squares reduction
    __shared__ float s_sum_sq;

    int tid = threadIdx.x;

    // Initialize shared memory on the first thread
    if (tid == 0) {
        s_sum_sq = 0.0f;
    }
    __syncthreads();

    // 2. Accumulate local sums of squares using a grid-stride loop across columns
    float local_sum_sq = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float val = input[row * N + col];
        local_sum_sq += val * val;
    }

    // 3. Atomically aggregate the block results into shared memory
    atomicAdd(&s_sum_sq, local_sum_sq);
    __syncthreads();

    // 4. Calculate the inverse Root-Mean-Square
    float mean_square = s_sum_sq / N;
    float inv_rms = rsqrtf(mean_square + eps);

    // 5. Normalize elements and apply the scale vector gamma
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        output[idx] = input[idx] * inv_rms * gamma[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}