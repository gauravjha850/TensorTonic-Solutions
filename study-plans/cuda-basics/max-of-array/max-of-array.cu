#include <cuda_runtime.h>
#include <float.h>

// Safe atomicMax implementation for floats using atomicCAS
__device__ inline void atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);
}

__global__ void max_kernel(const float* input, float* result, int N) {
    // Shared memory allocated per block (256 threads)
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory, filling out-of-bounds indices with -FLT_MAX
    if (idx < N) {
        s_max[tid] = input[idx];
    } else {
        s_max[tid] = -FLT_MAX;
    }
    __syncthreads();

    // Perform intra-block tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write the block's maximum to the global result safely
    if (tid == 0) {
        atomicMaxFloat(result, s_max[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Initialize result with negative infinity
    float neg_inf = -FLT_MAX;
    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch the reduction kernel
    max_kernel<<<blocks, threads>>>(input, result, N);
    
    cudaDeviceSynchronize();
}
