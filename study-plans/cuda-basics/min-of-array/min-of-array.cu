#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float *result) { 
    result[0] = FLT_MAX; 
}

// Safe atomicMin implementation for floats using atomicCAS
__device__ inline void atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(__int_as_float(assumed), val)));
    } while (assumed != old);
}

__global__ void min_kernel(const float *input, float *result, int N) {
    // Shared memory allocated per block (256 threads)
    __shared__ float s_min[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory; pad out-of-bounds with FLT_MAX
    if (idx < N) {
        s_min[tid] = input[idx];
    } else {
        s_min[tid] = FLT_MAX;
    }
    __syncthreads();

    // Perform intra-block tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
        }
        __syncthreads();
    }

    // Atomically update the global minimum from the first thread of each block
    if (tid == 0) {
        atomicMinFloat(result, s_min[0]);
    }
}

extern "C" void solve(const float *input, float *result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    init_result<<<1, 1>>>(result);
    min_kernel<<<blocks, threads>>>(input, result, N);
    
    cudaDeviceSynchronize();
}
