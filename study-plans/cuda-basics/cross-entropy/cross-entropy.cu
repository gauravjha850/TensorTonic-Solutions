#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Kernel 1: Computes the unnormalized cross-entropy loss contribution per row
__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* partial_loss, int B, int C) {
    // Each block processes exactly one row (one batch element)
    int row = blockIdx.x;
    if (row >= B) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Shared memory allocated for 256 threads
    __shared__ float s_mem[256];

    // --- Step 1: Compute Per-Row Max Logit (Grid-Stride Loop over columns) ---
    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += bdim) {
        local_max = fmaxf(local_max, logits[row * C + c]);
    }
    s_mem[tid] = local_max;
    __syncthreads();

    // Block-wide parallel tree reduction for the maximum value
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads(); // Reuse shared memory safely

    // --- Step 2: Compute Log-Sum-Exp Denominator ---
    float local_sum = 0.0f;
    for (int c = tid; c < C; c += bdim) {
        local_sum += expf(logits[row * C + c] - row_max);
    }
    s_mem[tid] = local_sum;
    __syncthreads();

    // Block-wide parallel tree reduction for the sum of exponentials
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    float log_sum_exp = logf(s_mem[0]);

    // --- Step 3: Compute Loss for the Target Class ---
    if (tid == 0) {
        int target_class = target[row];
        float target_logit = logits[row * C + target_class];
        
        // Negative Log-Softmax formula: -(z_target - z_max - lse)
        partial_loss[row] = -(target_logit - row_max - log_sum_exp);
    }
}

// Kernel 2: Aggregates the per-batch losses and normalizes by B
__global__ void cross_entropy_finalize_kernel(const float* partial_loss, float* loss, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_loss = 0.0f;
        for (int i = 0; i < B; ++i) {
            total_loss += partial_loss[i];
        }
        loss[0] = total_loss / B;
    }
}

// Host entry function
extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    // Allocate device memory for holding each row's individual loss contribution
    float* d_partial_loss = nullptr;
    cudaMalloc(&d_partial_loss, B * sizeof(float));

    // Configure a block with 256 threads to perform the reductions smoothly
    int threads = 256;
    int blocks = B; // One block per row

    // 1. Calculate loss per row with numerical stability tricks
    cross_entropy_row_kernel<<<blocks, threads>>>(logits, target, d_partial_loss, B, C);

    // 2. Compute the final average over the entire batch size
    cross_entropy_finalize_kernel<<<1, 1>>>(d_partial_loss, loss, B);

    // Synchronize execution and release temporary device storage
    cudaDeviceSynchronize();
    cudaFree(d_partial_loss);
}