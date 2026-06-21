#include <cuda_runtime.h>
#include <float.h>

// 1. CUDA Kernel to perform 2D max pooling (valid padding)
__global__ void max_pool_2d_kernel(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    // Calculate out dimensions
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    // Map thread indices to 2D output coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Output Column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Output Row

    // 2. Verify boundaries against the valid output shape
    if (i < outH && j < outW) {
        float max_val = -FLT_MAX;

        // Calculate the base starting point in the input tensor
        int start_r = i * sH;
        int start_c = j * sW;

        // 3. Scan the window pooling region
        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                int input_idx = (start_r + a) * W + (start_c + b);
                float val = input[input_idx];
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        // Write once to row-major output stride outW
        output[i * outW + j] = max_val;
    }
}

// Host entry function
extern "C" void solve(const float* input, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    // Configure 16x16 execution thread blocks as required
    dim3 threads(16, 16);

    // Calculate grid blocks needed to completely cover the output matrix dimensions
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);

    // Launch the Max Pool 2D kernel
    max_pool_2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW, sH, sW);

    // Synchronize host execution
    cudaDeviceSynchronize();
}

