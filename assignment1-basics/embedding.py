import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        weights = torch.empty(num_embeddings, embedding_dim, 
                            device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weights, std=1.0, a=-3.0, b=3.0)
        self.weights = torch.nn.Parameter(weights)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
