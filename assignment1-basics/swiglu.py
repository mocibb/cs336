import torch
from einops import einsum
from math import sqrt

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, weights=None, device=None, dtype=None):
        super().__init__()

        def init_param(shape):
                param = torch.empty(*shape, device=device, dtype=dtype)
                sigma = sqrt(2 / sum(shape))
                torch.nn.init.trunc_normal_(param, std=sigma, a=-3*sigma, b=3*sigma)
                return param

        self.w1 = torch.nn.Parameter(init_param((d_ff, d_model)))
        self.w2 = torch.nn.Parameter(init_param((d_model, d_ff)))
        self.w3 = torch.nn.Parameter(init_param((d_ff, d_model)))
        self.silu = torch.nn.SiLU()
        
    def forward(self, x):
        return einsum(self.w2, 
                      self.silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")) * 
                      einsum(self.w3, x, "d_ff m, ... m -> ... d_ff"),
                      "m d_ff, ... d_ff -> ... m")
        