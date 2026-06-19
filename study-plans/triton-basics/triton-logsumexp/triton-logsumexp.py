import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Map this program instance to its specific row
    row_idx = tl.program_id(0)
    
    # Calculate row pointer using the row stride
    row_start_ptr = x_ptr + row_idx * x_row_stride
    
    # Generate block offsets and mask for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 1. Load the raw row values into registers
    # Use -inf as padding so masked lanes don't affect the maximum value
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 2. Find the row maximum for numerical stability
    row_max = tl.max(x, axis=0)
    
    # 3. Shift and exponentiate 
    shifted_x = x - row_max
    numerator = tl.exp(shifted_x)
    
    # 4. Compute the sum of exponentials
    sum_exp = tl.sum(numerator, axis=0)
    
    # 5. Reconstruct the log-sum-exp value by adding back the row_max
    lse = row_max + tl.log(sum_exp)
    
    # 6. Store the scalar result back into the 1D output tensor for this row
    out_row_ptr = out_ptr + row_idx
    tl.store(out_row_ptr, lse)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M, )
    
    logsumexp_kernel[grid](
        x, out, 
        x.stride(0), 
        N, 
        BLOCK_SIZE=BLOCK_SIZE,
    )