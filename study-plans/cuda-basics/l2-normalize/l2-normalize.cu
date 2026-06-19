#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Multi-element reduction using a single block with a grid-stride loop
__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
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

    // Shared memory tree reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the total sum of squares to global memory
    if (tid == 0) {
        *sumv = s_data[0];
    }
}

// Kernel 2: Element-wise division using the fast inverse square root
__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Use rsqrtf for maximum hardware acceleration of 1 / sqrt(x)
        float inv_norm = rsqrtf(*sumv);
        output[i] = input[i] * inv_norm;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    // Reduce all squared values using 1 block of 256 threads
    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Normalize elements element-wise
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
