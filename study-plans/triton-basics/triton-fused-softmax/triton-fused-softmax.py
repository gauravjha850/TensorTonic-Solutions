import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, out_ptr, 
    x_row_stride, out_row_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    # Map this program instance to its specific row
    row_idx = tl.program_id(0)
    
    # Calculate row pointers using the row strides
    row_start_ptr = x_ptr + row_idx * x_row_stride
    out_start_ptr = out_ptr + row_idx * out_row_stride
    
    # Generate block offsets and mask for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 1. Load the raw row values into registers
    # Use -inf as padding so masked lanes don't affect the maximum value
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 2. Find the row maximum for numerical stability
    row_max = tl.max(x, axis=0)
    
    # 3. Shift and exponentiate 
    # Safe fallback of 0.0 for the exponential sum of masked lanes (exp(-inf) -> 0)
    shifted_x = x - row_max
    numerator = tl.exp(shifted_x)
    
    # 4. Compute the normalization denominator
    denominator = tl.sum(numerator, axis=0)
    
    # 5. Normalize and write the final result back to memory
    softmax_out = numerator / denominator
    tl.store(out_start_ptr + col_offsets, softmax_out, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M, )
    
    softmax_kernel[grid](
        x, out, 
        x.stride(0), out.stride(0), 
        N, 
        BLOCK_SIZE=BLOCK_SIZE,
    )