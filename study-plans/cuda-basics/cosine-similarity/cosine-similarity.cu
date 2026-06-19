#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_partials_kernel(const float *A, const float *B, float *scratch, int N) {
    __shared__ float s_dot[256];
    __shared__ float s_a2[256];
    __shared__ float s_b2[256];

    int tid = threadIdx.x;
    
    float local_dot = 0.0f;
    float local_a2 = 0.0f;
    float local_b2 = 0.0f;

    // Grid-stride loop to handle arbitrary sizes when blocks are capped at 1024
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float va = A[i];
        float vb = B[i];
        
        local_dot += va * vb;
        local_a2  += va * va;
        local_b2  += vb * vb;
    }

    s_dot[tid] = local_dot;
    s_a2[tid]  = local_a2;
    s_b2[tid]  = local_b2;
    __syncthreads();

    // Block-level parallel tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
            s_a2[tid]  += s_a2[tid + stride];
            s_b2[tid]  += s_b2[tid + stride];
        }
        __syncthreads();
    }

    // Atomically aggregate block totals into the 3-element scratch workspace
    if (tid == 0) {
        atomicAdd(&scratch[0], s_dot[0]);
        atomicAdd(&scratch[1], s_a2[0]);
        atomicAdd(&scratch[2], s_b2[0]);
    }
}

__global__ void cosine_finalize_kernel(const float *scratch, float *result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float dot_product = scratch[0];
        float norm_a2     = scratch[1];
        float norm_b2     = scratch[2];
        
        // Sim(A, B) = Dot(A, B) / (||A|| * ||B||)
        result[0] = dot_product / (sqrtf(norm_a2) * sqrtf(norm_b2));
    }
}

extern "C" void solve(const float *A, const float *B, float *result, int N) {
    float *scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    
    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);
    
    cudaDeviceSynchronize();
    cudaFree(scratch);
}