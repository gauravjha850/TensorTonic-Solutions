#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Shared memory allocations for reduction blocks
    __shared__ float s_max;
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // Initialize block-wide metrics on the first thread
    if (tid == 0) {
        s_max = -INFINITY;
        s_sum = 0.0f;
    }
    __syncthreads();

    // 1. Grid-stride loop to find the true maximum element
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        if (input[i] > local_max) {
            local_max = input[i];
        }
    }

    // Atomic max operation within shared memory (highly efficient)
    int* s_max_int = (int*)&s_max;
    int old = *s_max_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= local_max) break;
        old = atomicCAS(s_max_int, assumed, __float_as_int(local_max));
    } while (assumed != old);

    __syncthreads();

    // 2. Grid-stride loop to sum up the shifted exponents
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - s_max);
    }
    atomicAdd(&s_sum, local_sum);

    __syncthreads();

    // 3. Grid-stride loop to calculate final normalized probability values
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - s_max) / s_sum;
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    // Launching EXACTLY ONE block to guarantee mid-kernel block synchronization
    int threads = 256;
    softmax_kernel<<<1, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
