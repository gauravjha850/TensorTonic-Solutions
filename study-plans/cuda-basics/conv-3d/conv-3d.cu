#include <cuda_runtime.h>

// 1. CUDA Kernel to perform 3D cross-correlation (valid padding, stride 1)
__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, 
                              int D, int H, int W, int kD, int kH, int kW) {
    // Calculate output volume dimensions
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Map the 3D thread grid to the output volume coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Width index (X)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Height index (Y)
    int d = blockIdx.z * blockDim.z + threadIdx.z; // Depth index (Z)

    // 2. Ensure the thread falls within the valid output boundaries
    if (d < outD && i < outH && j < outW) {
        float sum = 0.0f;

        // 3. Perform the 3D cross-correlation sliding window loop over (kD, kH, kW)
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kH; ++b) {
                for (int c = 0; c < kW; ++c) {
                    // Flattened indexing for row-major layouts:
                    // input: (d + a) * H * W + (i + b) * W + (j + c)
                    int input_idx = (d + a) * H * W + (i + b) * W + (j + c);
                    
                    // kernel: a * kH * kW + b * kW + c
                    int kernel_idx = a * kH * kW + b * kW + c;

                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }

        // Write the finalized accumulation to the flattened row-major output tensor
        int output_idx = d * outH * outW + i * outW + j;
        output[output_idx] = sum;
    }
}

// Host entry function
extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int D, int H, int W, int kD, int kH, int kW) {
    // Calculate output volume bounding dimensions
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Configure a 3D thread block (8x8x8 = 512 threads per block, well under the 1024 limit)
    dim3 threads(8, 8, 8);

    // Calculate a 3D grid layout to fully encapsulate the target output shape
    dim3 blocks((outW + 7) / 8, 
                (outH + 7) / 8, 
                (outD + 7) / 8);

    // Launch the core 3D convolution kernel
    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);

    // Wait for the device pipeline stream to finalize execution
    cudaDeviceSynchronize();
}
