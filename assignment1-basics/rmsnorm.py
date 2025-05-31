import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms = \sqrt{\frac{1}{N} \sum_{i=1}^{N} a_i^2+\varepsilon}
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # result = x_i * \frac{w_i}{rms}
        return (x*self.weights/rms).to(in_dtype)