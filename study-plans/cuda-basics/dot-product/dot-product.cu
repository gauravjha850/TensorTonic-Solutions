#include <cuda_runtime.h>

// 1. CUDA Kernel to perform dot product reduction
__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    // Shared memory size must match the block size configured below
    __shared__ float s_data[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Accumulate local products via grid-stride loop
    float local_prod = 0.0f;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        local_prod += A[idx] * B[idx];
    }
    
    // Write partial thread sum to shared memory
    s_data[tid] = local_prod;
    __syncthreads();
    
    // 3. Tree reduction inside shared memory for this block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    // 4. Atomically add this block's total to the global result vector
    if (tid == 0) {
        atomicAdd(result, s_data[0]);
    }
}

// Host entry function
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    
    // Calculate block allocation to sufficiently tile across N elements
    int blocks = (N + threads - 1) / threads;
    
    // Clamp the grid size to avoid launching excess idle blocks on large N
    if (blocks > 1024) {
        blocks = 1024;
    }
    
    // Launch the dot product reduction kernel
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    
    // Wait for GPU execution to finalize
    cudaDeviceSynchronize();
}
