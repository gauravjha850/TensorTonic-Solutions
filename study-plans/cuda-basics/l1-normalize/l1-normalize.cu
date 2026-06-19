#include <cuda_runtime.h>
#include <math.h>

// Pass 1: Shared memory tree reduction to sum absolute values
__global__ void l1_sum_kernel(const float* input, float* global_sum, int N) {
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_val = 0.0f;
    if (i < N) {
        local_val = fabsf(input[i]);
    }
    s_data[tid] = local_val;
    __syncthreads();

    // Parallel tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate block total into global memory
    if (tid == 0) {
        atomicAdd(global_sum, s_data[0]);
    }
}

// Pass 2: Element-wise division by the total absolute sum
__global__ void l1_divide_kernel(const float* input, float* output, const float* global_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] / (*global_sum);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Allocate scratch device space for the global absolute sum
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    // Step 1: Compute denominator
    l1_sum_kernel<<<blocks, threads>>>(input, d_sum, N);
    
    // Step 2: Divide each element by denominator
    l1_divide_kernel<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    
    // Clean up temporary allocation
    cudaFree(d_sum);
}
