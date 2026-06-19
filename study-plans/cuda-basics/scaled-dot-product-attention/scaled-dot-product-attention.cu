#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Step 1: Compute the scaled score matrix: S = (Q * K^T) / sqrt(D)
__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of S (maps to row of K)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of S (maps to row of Q)

    if (i < N && j < N) {
        float sum = 0.0f;
        // Compute dot product between Q[i] and K[j]
        for (int d = 0; d < D; ++d) {
            sum += Q[i * D + d] * K[j * D + d];
        }
        
        float scale = 1.0f / sqrtf((float)D);
        scores[i * N + j] = sum * scale;
    }
}

// Step 2: Numerically stable row-wise softmax over the scores matrix (shape N x N)
__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x; // Each block processes exactly one row
    if (row >= N) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ float s_mem[256];

    // 2a. Find the maximum element in the row
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += bdim) {
        local_max = fmaxf(local_max, scores[row * N + j]);
    }
    s_mem[tid] = local_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads();

    // 2b. Compute sum of exponentials (with max subtraction)
    float local_sum_exp = 0.0f;
    for (int j = tid; j < N; j += bdim) {
        local_sum_exp += expf(scores[row * N + j] - row_max);
    }
    s_mem[tid] = local_sum_exp;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_mem[0];
    __syncthreads();

    // 2c. Write out normalized softmax probabilities back to scores
    for (int j = tid; j < N; j += bdim) {
        scores[row * N + j] = expf(scores[row * N + j] - row_max) / row_sum;
    }
}

// Step 3: Value aggregation: output = attn * V
__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index of output/V
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of output/attn

    if (i < N && d < D) {
        float sum = 0.0f;
        // Dot product between row i of attn (probabilities) and column d of V
        for (int j = 0; j < N; ++j) {
            sum += attn[i * N + j] * V[j * D + d];
        }
        output[i * D + d] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));
    
    // Matrix Multiplication Q @ K.T
    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);
    
    // Row-wise Softmax over Scores
    softmax_rows_kernel<<<N, 256>>>(scores, N);
    
    // Weighted Value Matrix Aggregation
    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);
    
    cudaDeviceSynchronize();
    cudaFree(scores);
}