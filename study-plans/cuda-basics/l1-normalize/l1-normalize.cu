#include <cuda_runtime.h>
#include <math.h>

// Pass 1: Shared memory tree reduction using grid-stride loop to sum absolute values
__global__ void l1_sum_kernel(const float* input, float* global_sum, int N) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to comfortably handle N up to 10^7
    float local_sum = 0.0f;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        local_sum += fabsf(input[idx]);
    }
    
    s_data[tid] = local_sum;
    __syncthreads();
    
    // Parallel tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Accumulate the block total into the global scratch variable
    if (tid == 0) {
        atomicAdd(global_sum, s_data[0]);
    }
}

// Pass 2: Element-wise normalization divide pass
__global__ void l1_normalize_kernel(const float* input, float* output, const float* global_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Read the absolute sum computed by the first pass
        output[i] = input[i] / (*global_sum);
    }
}

// Host entry function
extern "C" void solve(const float* input, float* output, int N) {
    // Allocate a temporary device pointer to accumulate the denominator
    float* d_global_sum = nullptr;
    cudaMalloc(&d_global_sum, sizeof(float));
    cudaMemset(d_global_sum, 0, sizeof(float));
    
    int threads = 256;
    // Cap grid size to optimize wave scheduling on huge N
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    // 1. Compute the global L1 norm denominator
    l1_sum_kernel<<<blocks, threads>>>(input, d_global_sum, N);
    
    // 2. Perform the element-wise scaling transformation
    int norm_blocks = (N + threads - 1) / threads;
    l1_normalize_kernel<<<norm_blocks, threads>>>(input, output, d_global_sum, N);
    
    // Synchronize host execution and clean up scratch allocation
    cudaDeviceSynchronize();
    cudaFree(d_global_sum);
}
