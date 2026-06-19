#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to protect against out-of-bounds memory accesses
    if (idx < N) {
        // Precompute inverted dropout scaling factor
        float scale = 1.0f / (1.0f - p);
        
        // Inverted dropout formula: (input * mask) / (1 - p)
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    
    cudaDeviceSynchronize();
}