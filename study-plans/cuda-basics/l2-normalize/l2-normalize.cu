#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Multi-element reduction using a single block with a grid-stride loop
__global__ void reduce_sq_sum(const float* input, float* global_sq_sum, int N) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;
    
    float local_sq_sum = 0.0f;
    // Grid-stride loop allowing 256 threads to cover up to 10^7 elements safely
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[i];
        local_sq_sum += val * val;
    }
    
    s_data[tid] = local_sq_sum;
    __syncthreads();
    
    // Shared memory parallel tree reduction within the single block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the block's total sum of squares to global memory
    if (tid == 0) {
        *global_sq_sum = s_data[0];
    }
}

// Kernel 2: Element-wise normalization pass
__global__ void normalize_kernel(const float* input, float* output, const float* global_sq_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        // Fast reciprocal square root: rsqrtf(x) calculates 1 / sqrt(x)
        float inv_norm = rsqrtf(*global_sq_sum);
        output[i] = input[i] * inv_norm;
    }
}

// Host entry function
extern "C" void solve(const float* input, float* output, int N) {
    // 1. Allocate a single-float scratch buffer on the device
    float* d_sq_sum = nullptr;
    cudaMalloc(&d_sq_sum, sizeof(float));
    cudaMemset(d_sq_sum, 0, sizeof(float));
    
    // 2. Launch reduction kernel with a single block of 256 threads
    reduce_sq_sum<<<1, 256>>>(input, d_sq_sum, N);
    
    // 3. Launch normalization kernel with standard 1D grid layout
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(input, output, d_sq_sum, N);
    
    // 4. Synchronize device execution and clean up scratch allocation
    cudaDeviceSynchronize();
    cudaFree(d_sq_sum);
}
