#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    // Map the 3D thread grid to the output volume coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Width index (X)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Height index (Y)
    int d = blockIdx.z * blockDim.z + threadIdx.z; // Depth index (Z)

    // Ensure the thread falls within the valid output boundaries
    if (d < outD && i < outH && j < outW) {
        float sum = 0.0f;

        // Perform the 3D cross-correlation sliding window loop
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kH; ++b) {
                for (int c = 0; c < kW; ++c) {
                    // Flatten 3D coordinates into row-major 1D indices
                    int input_idx = (d + a) * (H * W) + (i + b) * W + (j + c);
                    int kernel_idx = a * (kH * kW) + b * kW + c;

                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }

        // Write the final reduction value out to global memory
        int output_idx = d * (outH * outW) + i * outW + j;
        output[output_idx] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    
    // Configured with the required 3D layout (8, 8, 8)
    dim3 threads(8, 8, 8);
    dim3 blocks((outW + 7) / 8, (outH + 7) / 8, (outD + 7) / 8);
    
    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
    
    cudaDeviceSynchronize();
}
