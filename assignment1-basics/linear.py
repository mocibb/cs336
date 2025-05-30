import torch
from math import sqrt
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()
                
        if weights is not None:
            self.weights = torch.nn.Parameter(weights)
        else:
            weights = torch.empty(out_features, in_features, 
                                device=device, dtype=dtype)
            sigma = sqrt(2/(in_features+out_features))
            torch.nn.init.trunc_normal_(weights, std=sqrt(2./(in_features+out_features)), a=-3.0*sigma, b=3.0*sigma)
            self.weights = torch.nn.Parameter(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = W x
        return einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")
    