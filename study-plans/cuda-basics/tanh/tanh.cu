#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // 1. Calculate the global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. Perform the Tanh operation if within bounds
    if (i < N) {
        output[i] = tanhf(input[i]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Launch the renamed kernel
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    
    cudaDeviceSynchronize();
}