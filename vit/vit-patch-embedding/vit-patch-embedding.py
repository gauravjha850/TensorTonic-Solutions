import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    B,H,W,C=image.shape
    P=patch_size
    num_h=H//P
    num_w=W//P
    image_cropped=image[:,:num_h*P,:num_w*P,:]
    patches=image_cropped.reshape(B,num_h,P,num_w,P,C).transpose(0,1,3,2,4,5)
    x_p=patches.reshape(B,num_h*num_w,P*P*C)
    zp=(np.matmul(x_p,W_proj)+bias)
    return zp.astype(np.float64)
    
    
    