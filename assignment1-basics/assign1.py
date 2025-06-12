from train_bpe import train_bpe
# from train_bpe_slow import train_bpe
from tokenizer import Tokenizer
from model import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, MultiheadSelfAttention, TransformerBlock, TransformerBlockLM
from model import softmax, scaled_dot_product_attention
from optimizer import cross_entropy, gradient_clipping, get_lr_cosine_schedule, AdamW
from training import get_batch, save_checkpoint, load_checkpoint
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
           'gradient_clipping',
           'get_lr_cosine_schedule',
           'AdamW',
           'get_batch',
           'save_checkpoint',
           'load_checkpoint'
           ]

def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x * torch.sigmoid(x)
