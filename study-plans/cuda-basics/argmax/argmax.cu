#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Shared memory arrays to track values and indices per block
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements. Out-of-bounds items are padded with -FLT_MAX and a dummy index.
    if (idx < N) {
        s_vals[tid] = input[idx];
        s_idxs[tid] = idx;
    } else {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = idx; // Keeps index ordered safely, though values won't compete
    }
    __syncthreads();

    // Perform tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float val1 = s_vals[tid];
            float val2 = s_vals[tid + stride];
            
            // Tie-break rule: keep larger value. If equal, keep the smaller index.
            if (val2 > val1) {
                s_vals[tid] = val2;
                s_idxs[tid] = s_idxs[tid + stride];
            } else if (val2 == val1) {
                if (s_idxs[tid + stride] < s_idxs[tid]) {
                    s_idxs[tid] = s_idxs[tid + stride];
                }
            }
        }
        __syncthreads();
    }

    // Write the block's winning pair to the global scratch workspace
    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;

    // Initialize shared memory with identity values
    float final_val = -FLT_MAX;
    int final_idx = INT_MAX;

    // Linearly loop over block aggregates if num_blocks > 256 (strided loop)
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];
        
        if (val > final_val) {
            final_val = val;
            final_idx = idx;
        } else if (val == final_val) {
            if (idx < final_idx) {
                final_idx = idx;
            }
        }
    }

    s_vals[tid] = final_val;
    s_idxs[tid] = final_idx;
    __syncthreads();

    // Block-level reduction on the collected sub-results
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float val1 = s_vals[tid];
            float val2 = s_vals[tid + stride];
            
            if (val2 > val1) {
                s_vals[tid] = val2;
                s_idxs[tid] = s_idxs[tid + stride];
            } else if (val2 == val1) {
                if (s_idxs[tid + stride] < s_idxs[tid]) {
                    s_idxs[tid] = s_idxs[tid + stride];
                }
            }
        }
        __syncthreads();
    }

    // Write the final global minimum index to the output location
    if (tid == 0) {
        result[0] = s_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));
    
    // Step 1: Find local maximum and index per thread block
    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    
    // Step 2: Finalize the global maximum index across all blocks
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    
    cudaDeviceSynchronize();
    
    cudaFree(block_vals);
    cudaFree(block_idxs);
}
