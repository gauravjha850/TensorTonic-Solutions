#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    // Allocate shared memory for the block-wide reduction
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Fetch element from global memory (with bounds checking)
    float local_sum = 0.0f;
    if (i < N) {
        local_sum = input[i];
    }
    s_data[tid] = local_sum;
    __syncthreads();

    // 2. Perform parallel tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // 3. Thread 0 writes this block's total to the shared global result
    if (tid == 0) {
        atomicAdd(result, s_data[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Clear out target memory buffer allocation before aggregating values
    cudaMemset(result, 0, sizeof(float));
    
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
