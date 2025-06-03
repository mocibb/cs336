import torch
from math import sqrt
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
                
        weight = torch.empty(in_features, out_features, 
                             device=device, dtype=dtype)
        sigma = sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(weight, std=sigma, a=-3.0*sigma, b=3.0*sigma)
        self.weight = torch.nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = W x
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        weight = torch.empty(num_embeddings, embedding_dim, 
                             device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, std=1.0, a=-3.0, b=3.0)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms = \sqrt{\frac{1}{N} \sum_{i=1}^{N} a_i^2+\varepsilon}
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # result = x_i * \frac{w_i}{rms}
        return (x*self.weight/rms).to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_ff, d_model)
        self.w2 = Linear(d_model, d_ff)
        self.w3 = Linear(d_ff, d_model)
        self.silu = lambda x: x * torch.sigmoid(x)
        
    def forward(self, x):
        return self.w2(self.silu(self.w1(x))*self.w3(x))
        

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

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_vals = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_vals 
    exp_vals = torch.exp(shifted)
    return exp_vals / exp_vals.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: Float[Tensor, "... n d_k"],
                                 K: Float[Tensor, "... m d_k"],
                                 V: Float[Tensor, "... m d_v"],
                                 mask: Float[Tensor, "seq_len seq_len"] | None = None ) -> Float[Tensor, "... d_v"]:
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf) 
    
    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, V, "... n m, ... m d_v -> ... n d_v")
    

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                 max_seq_len: int | None = None, 
                 theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None, apply_rope: bool = False):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.apply_rope = apply_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.apply_rope:
            self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device=device)
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(-2)

        Q, K, V = (
            rearrange(X(x),
                    "... s (h d) -> ... h s d", h=self.num_heads)
                    for X in (self.q_proj, self.k_proj, self.v_proj)
        )  

        if self.rope is not None:
            positions = rearrange(torch.arange(seq_len, device=x.device), "seq -> 1 seq").expand(batch_size, -1)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).expand(batch_size, self.num_heads, -1, -1)
        attn_output = rearrange(scaled_dot_product_attention(Q, K, V, mask), " ... h s d_v -> ... s (h d_v)")
        
        return self.output_proj(attn_output)
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None):
        super().__init__()

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, apply_rope=True)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:       
        x = self.ln1(in_features)
        x = self.attn(x)
        y = in_features + x
        return y + self.ffn(self.ln2(y))