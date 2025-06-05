import torch
from math import sqrt
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(inputs: Float[Tensor, "b v"], targets: Int[Tensor, "b"]) -> Float[Tensor, ""]:
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values 
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(1, targets.unsqueeze(1))
    return torch.mean(log_sum_exp-target_logits)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                return loss
            

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                t, m, v = state['t'], state['m'], state['v']
                # 这里不申请新的内存
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # scalar
                lr_t = lr * sqrt(1-beta2**t) / (1-beta1**t)
                denom = v.sqrt().add_(eps)
                # ltheta -= lr_t * m / denom
                p.data.addcdiv_(m, denom, value=-lr_t)
                # theta -= lr * wd * theta
                p.data.mul_(1 - lr * wd)

                state["t"] = t + 1
        
        return loss