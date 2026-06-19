#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Map thread indices to 2D output coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index (X-dimension)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (Y-dimension)

    // Check boundaries against the valid output shape
    if (i < outH && j < outW) {
        float sum = 0.0f;
        
        // Loop over the 2D window of the kernel
        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                // Map 2D indices to flat row-major indices
                int input_idx = (i + a) * W + (j + b);
                int kernel_idx = a * kW + b;
                
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
        
        // Write the final accumulated dot product to the output array
        output[i * outW + j] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    
    // Configure execution configuration as requested
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    
    cudaDeviceSynchronize();
}
