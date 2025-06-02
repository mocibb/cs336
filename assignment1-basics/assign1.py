from train_bpe import train_bpe
from tokenizer import Tokenizer
from linear import Linear
from embedding import Embedding
from rmsnorm import RMSNorm
from swiglu import SwiGLU
from rope import RotaryPositionalEmbedding
import torch
from torch import Tensor
from math import sqrt
from jaxtyping import Float, Int
from einops import einsum, rearrange
import torch.nn.functional as F
import einx

__all__ = ['train_bpe', 
           'Tokenizer', 
           'Linear', 
           'Embedding', 
           'RMSNorm', 
           'SwiGLU', 
           'RotaryPositionalEmbedding', 
           'silu',
           'softmax',
           'scaled_dot_product_attention',
           'multihead_self_attention'
           ]

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)

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
    

def multihead_self_attention(d_model: int,
                             num_heads: int,
                             q_proj_weight: Float[Tensor, "hd_k d_model"],
                             k_proj_weight: Float[Tensor, "hd_k d_model"],
                             v_proj_weight: Float[Tensor, "hd_v d_model"],
                             o_proj_weight: Float[Tensor, "d_model hd_v"],
                             in_features: Float[Tensor, "... seq_len d_in"]) -> Float[Tensor, " ... seq_len d_model"]:
    batch_size = in_features.size(0)
    seq_len = in_features.size(-2)
    
    Q_heads, K_heads, V_heads = (
        rearrange(einsum(X, in_features, "hd d_model, ... s d_model -> ... s hd"),
                  "... s (h d) -> ... h s d", h=num_heads)
                  for X in (q_proj_weight, k_proj_weight, v_proj_weight)
    )  
    # query x key 维矩阵，key i只可以访问 j<=i的数据， 所以(j, i) = True, if j <= i 
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device)).expand(batch_size, num_heads, -1, -1)
    attn_output = rearrange(scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask), " ... h s d_v -> ... s (h d_v)")
    return einsum(o_proj_weight, attn_output, "d_model hd_v, ... hd_v -> ... d_model")


