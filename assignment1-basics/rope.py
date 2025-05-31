
import torch
from einops import rearrange
from math import pi

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        half_dim = d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        
        # 缓存三角函数表
        idx = torch.arange(max_seq_len, device=device)
        theta_table : Float[Tensor, "max_seq_len half_dim"] = torch.outer(idx, inv_freq) 
        self.register_buffer('sin', theta_table.sin(), persistent=False)
        self.register_buffer('cos', theta_table.cos(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 获取对应位置的旋转矩阵
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        # x1 = x[..., 0::2]
        # x2 = x[..., 1::2]
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        # torch.stack(..., dim=-1) 在新的维度上叠加
        # torch.flatten(..., start_dim=-2) 将最后两个维度合并
        return torch.stack((x1*cos-x2*sin, x1*sin+x2*cos), dim=-1).flatten(start_dim=-2)