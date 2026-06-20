#include <cuda_runtime.h>

// 1. CUDA Kernel to perform 2D cross-correlation (valid padding, stride 1)
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    // Calculate output matrix boundary constraints
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Map thread indices to 2D output matrix coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index (X-dimension)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (Y-dimension)

    // 2. Guard against out-of-bounds execution inside the output matrix space
    if (i < outH && j < outW) {
        float sum = 0.0f;

        // 3. Loop over the 2D window of the kernel sliding window footprint
        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                // Map the multi-dimensional offsets down to row-major linear boundaries
                int input_idx = (i + a) * W + (j + b);
                int kernel_idx = a * kW + b;

                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        // Write out the completed accumulation element to the final destination vector
        output[i * outW + j] = sum;
    }
}

// Host entry function
extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Configure a 16x16 execution thread block as required
    dim3 threads(16, 16);

    // Calculate grid blocks needed to completely encompass the modified target space
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);

    // Launch the 2D Convolution execution kernel
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);

    // Wait for the GPU pipeline execution tracking timeline to finish
    cudaDeviceSynchronize();
}
