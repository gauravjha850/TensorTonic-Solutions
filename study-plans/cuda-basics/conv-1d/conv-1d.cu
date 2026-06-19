#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds since the last block might have inactive threads
    if (idx < outN) {
        float sum = 0.0f;
        
        // Compute the dot-product for the window starting at input[idx]
        for (int a = 0; a < kN; ++a) {
            sum += input[idx + a] * kernel[a];
        }
        
        // Write the final accumulated result to global memory
        output[idx] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    
    cudaDeviceSynchronize();
}
