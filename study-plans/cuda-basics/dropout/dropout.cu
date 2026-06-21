#include <cuda_runtime.h>

// 1. CUDA Kernel to perform inverted dropout scaling
__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    // Calculate the unique global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Guard to protect against out-of-bounds memory accesses
    if (idx < N) {
        // Precompute the inverted dropout scaling factor
        float scale = 1.0f / (1.0f - p);
        
        // Inverted dropout formula: output = input * mask * scale
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

// Host entry function
extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    // Configure a standard 1D grid layout with 256 threads per block
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Launch the dropout kernel
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    
    // Synchronize the host with the device execution pipeline
    cudaDeviceSynchronize();
}