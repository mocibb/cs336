from train_bpe import train_bpe
from tokenizer import Tokenizer
from model import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, MultiheadSelfAttention, TransformerBlock, TransformerBlockLM
from model import softmax, scaled_dot_product_attention
from optimizer import cross_entropy, AdamW
import torch
from torch import Tensor
from jaxtyping import Float


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
           'MultiheadSelfAttention',
           'TransformerBlock',
           'TransformerBlockLM',
           'cross_entropy',
           'AdamW'
           ]

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)
