#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* partial, int B, int C) {
    // Each block processes exactly one row (one batch element)
    int row = blockIdx.x;
    if (row >= B) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ float s_mem[256];

    // --- Step 1: Compute Per-Row Max Logit (Strided Grid-Stride Style Loop for C) ---
    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += bdim) {
        local_max = fmaxf(local_max, logits[row * C + c]);
    }
    s_mem[tid] = local_max;
    __syncthreads();

    // Block reduction for Maximum
    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads();

    // --- Step 2: Compute Per-Row Sum of Exponentials ---
    float local_sum_exp = 0.0f;
    for (int c = tid; c < C; c += bdim) {
        local_sum_exp += expf(logits[row * C + c] - row_max);
    }
    s_mem[tid] = local_sum_exp;
    __syncthreads();

    // Block reduction for Sum-Exp
    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    float log_sum_exp = logf(s_mem[0]);

    // --- Step 3: Compute Cross-Entropy Loss for this Row ---
    if (tid == 0) {
        int t = target[row];
        float target_logit = logits[row * C + t];
        
        // Log-Softmax: z_t - row_max - log_sum_exp
        // Cross-Entropy: -Log-Softmax
        float row_loss = log_sum_exp - (target_logit - row_max);
        
        // Atomically accumulate to the global partial sum
        atomicAdd(partial, row_loss);
    }
}

__global__ void cross_entropy_finalize_kernel(float* loss, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        loss[0] = loss[0] / B;
    }
}

extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));
    
    // Each row of the batch gets its own block
    int threads = 256;
    dim3 blocks(B);
    
    cross_entropy_row_kernel<<<blocks, threads>>>(logits, target, loss, B, C);
    
    // Finalize step divides the global accumulated loss by the batch size B
    cross_entropy_finalize_kernel<<<1, 1>>>(loss, B);
    
    cudaDeviceSynchronize();
}