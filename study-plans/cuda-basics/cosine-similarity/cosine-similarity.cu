#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Multi-reduction grid-stride loop to gather partial dot and norm sums
__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    __shared__ float s_dot[256];
    __shared__ float s_a2[256];
    __shared__ float s_b2[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_dot = 0.0f;
    float local_a2 = 0.0f;
    float local_b2 = 0.0f;

    // Grid-stride loop to comfortably process elements up to N = 10^7
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        float va = A[idx];
        float vb = B[idx];
        
        local_dot += va * vb;
        local_a2  += va * va;
        local_b2  += vb * vb;
    }

    s_dot[tid] = local_dot;
    s_a2[tid]  = local_a2;
    s_b2[tid]  = local_b2;
    __syncthreads();

    // Parallel tree reduction within the shared memory arrays
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_a2[tid]  += s_a2[tid + stride];
            s_b2[tid]  += s_b2[tid + stride];
        }
        __syncthreads();
    }

    // Accumulate this block's final sums into global scratch slots
    if (tid == 0) {
        atomicAdd(&scratch[0], s_dot[0]);
        atomicAdd(&scratch[1], s_a2[0]);
        atomicAdd(&scratch[2], s_b2[0]);
    }
}

// Kernel 2: Single-threaded finalization pass to compute similarity division
__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float dot_prod = scratch[0];
        float norm_a2  = scratch[1];
        float norm_b2  = scratch[2];
        
        // Final Cosine Similarity computation: dot / (sqrt(a2) * sqrt(b2))
        result[0] = dot_prod / (sqrtf(norm_a2) * sqrtf(norm_b2));
    }
}

// Host entry function
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    // Allocate a 3-element scratch array on the device: [dot, sum_a2, sum_b2]
    float* d_scratch = nullptr;
    cudaMalloc(&d_scratch, 3 * sizeof(float));
    cudaMemset(d_scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Cap the grid size for massive inputs

    // 1. Gather all intermediate sums in a parallel single pass
    cosine_partials_kernel<<<blocks, threads>>>(A, B, d_scratch, N);

    // 2. Compute final square roots and division scalar
    cosine_finalize_kernel<<<1, 1>>>(d_scratch, result);

    // Synchronize host execution and clean up allocations
    cudaDeviceSynchronize();
    cudaFree(d_scratch);
}