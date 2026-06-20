#include <cuda_runtime.h>
#include <float.h>

// Kernel 1: Multi-block grid-stride reduction to find per-block partial minima
__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Shared memory arrays to track values and indices per block
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride initialization with default high padding
    float local_min = FLT_MAX;
    int local_idx = -1;
    
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        float val = input[idx];
        // Keep the smaller value; resolve ties with the lower index
        if (val < local_min || (val == local_min && (local_idx == -1 || idx < local_idx))) {
            local_min = val;
            local_idx = idx;
        }
    }
    
    s_vals[tid] = local_min;
    s_idxs[tid] = local_idx;
    __syncthreads();
    
    // In-block shared memory tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float remote_val = s_vals[tid + stride];
            int remote_idx = s_idxs[tid + stride];
            
            if (remote_val < s_vals[tid] || 
               (remote_val == s_vals[tid] && remote_idx < s_idxs[tid] && remote_idx != -1)) {
                s_vals[tid] = remote_val;
                s_idxs[tid] = remote_idx;
            }
        }
        __syncthreads();
    }
    
    // Write out the minimum winner of this block to global scratch arrays
    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

// Kernel 2: Single-block resolution pass to settle the absolute winning index
__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];
    
    int tid = threadIdx.x;
    
    float local_min = FLT_MAX;
    int local_idx = -1;
    
    // Scan across the block-level summary records
    for (int idx = tid; idx < num_blocks; idx += blockDim.x) {
        float val = block_vals[idx];
        int b_idx = block_idxs[idx];
        if (val < local_min || (val == local_min && (local_idx == -1 || b_idx < local_idx))) {
            local_min = val;
            local_idx = b_idx;
        }
    }
    
    s_vals[tid] = local_min;
    s_idxs[tid] = local_idx;
    __syncthreads();
    
    // Final tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float remote_val = s_vals[tid + stride];
            int remote_idx = s_idxs[tid + stride];
            
            if (remote_val < s_vals[tid] || 
               (remote_val == s_vals[tid] && remote_idx < s_idxs[tid] && remote_idx != -1)) {
                s_vals[tid] = remote_val;
                s_idxs[tid] = remote_idx;
            }
        }
        __syncthreads();
    }
    
    // Save the ultimate absolute minimum index to the final result buffer
    if (tid == 0) {
        result[0] = s_idxs[0];
    }
}

// Host entry function
extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Cap grid size safely to keep scratch arrays bounded
    
    // Allocate global device scratch arrays to hold intermediate block results
    float* d_block_vals = nullptr;
    int* d_block_idxs = nullptr;
    cudaMalloc(&d_block_vals, blocks * sizeof(float));
    cudaMalloc(&d_block_idxs, blocks * sizeof(int));
    
    // 1. First Pass: Compute minimum pairs across the full input vector
    argmin_kernel<<<blocks, threads>>>(input, d_block_vals, d_block_idxs, N);
    
    // 2. Second Pass: Reduce block records down using a single block
    argmin_finalize_kernel<<<1, threads>>>(d_block_vals, d_block_idxs, result, blocks);
    
    // 3. Synchronize stream execution and release scratch buffers
    cudaDeviceSynchronize();
    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
}
