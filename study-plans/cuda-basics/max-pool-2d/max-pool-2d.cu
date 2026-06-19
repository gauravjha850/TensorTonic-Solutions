#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    // Map thread indices to 2D output coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output Column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Output Row

    // Verify boundaries against the valid output shape
    if (i < outH && j < outW) {
        float max_val = -FLT_MAX;

        // Calculate the base starting point in the input tensor
        int start_r = i * sH;
        int start_c = j * sW;

        // Scan the window pooling region
        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                int input_idx = (start_r + a) * W + (start_c + b);
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }

        // Write the local maximum to the output array
        output[i * outW + j] = max_val;
    }
}

extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;
    
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    
    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);
    
    cudaDeviceSynchronize();
}

