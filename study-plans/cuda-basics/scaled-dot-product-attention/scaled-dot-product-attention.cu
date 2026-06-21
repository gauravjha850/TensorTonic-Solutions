#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// 1. Compute Scaled Dot-Product Scores: Scores = (Q @ K^T) / sqrt(D)
__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index of Scores (Key row)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of Scores (Query row)

    if (i < N && j < N) {
        float sum = 0.0f;
        // Compute dot product between Q[i] and K[j]
        for (int d = 0; d < D; ++d) {
            sum += Q[i * D + d] * K[j * D + d];
        }
        // Scale factor: 1.0f / sqrt(D)
        float scale = 1.0f / sqrtf((float)D);
        scores[i * N + j] = sum * scale;
    }
}

// 2. Numerically stable row-wise softmax over the Scores matrix
__global__ void softmax_rows_kernel(float* scores, int N) {
    int row = blockIdx.x; // One block handles one row of the N x N matrix
    if (row >= N) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ float s_mem[256];

    // Step 2a: Find max logit in the row (Grid-stride loop over columns)
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += bdim) {
        local_max = fmaxf(local_max, scores[row * N + j]);
    }
    s_mem[tid] = local_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads(); // Clear memory barrier

    // Step 2b: Sum of exponentials (denominator)
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += bdim) {
        local_sum += expf(scores[row * N + j] - row_max);
    }
    s_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_mem[0];

    // Step 2c: Write back the normalized softmax probabilities in-place
    for (int j = tid; j < N; j += bdim) {
        scores[row * N + j] = expf(scores[row * N + j] - row_max) / row_sum;
    }
}

// 3. Value aggregation matrix multiplication: Output = Attention_Weights @ V
__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Column index of output/V
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index of output/attn

    if (i < N && d < D) {
        float sum = 0.0f;
        // Dot product between row i of attn and column d of V
        for (int j = 0; j < N; ++j) {
            sum += attn[i * N + j] * V[j * D + d];
        }
        output[i * D + d] = sum;
    }
}

// Host entry function
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    // Dynamically allocate an N x N intermediate scratch matrix for the attention weights
    float* scores = nullptr;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    // Phase 1: Matrix Multiplication for Raw Scaled Scores (Q @ K.T / sqrt(D))
    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    // Phase 2: Row-wise Stable Softmax over intermediate scores
    // Maps 1 block per row with 256 reduction threads
    softmax_rows_kernel<<<N, 256>>>(scores, N);

    // Phase 3: Weighted Matrix Multiplication Aggregation (Attn @ V)
    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    // Synchronize execution stream and free the allocated scratch buffer
    cudaDeviceSynchronize();
    cudaFree(scores);
}