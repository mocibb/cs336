# coding=utf-8
# Copyright (c) 2025 mocibb (mocibb@163.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from math import sqrt, cos, pi
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(inputs: Float[Tensor, "b v"], targets: Int[Tensor, "b"]) -> Float[Tensor, ""]:
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values 
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(-1, targets.unsqueeze(1))
    return torch.mean(log_sum_exp-target_logits)

def gradient_clipping(param: Iterable[torch.nn.Parameter], M: float) -> None:
    grads = [p.grad.flatten() for p in param if p is not None and p.grad is not None]
    all_grads = torch.cat(grads, dim=0)
    norm = torch.norm(all_grads)

    if norm > M:
        clip_coef = M / (norm + 1.e-6)
        for p in param:
            if p is not None and p.grad is not None:
                p.grad *= clip_coef

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5*(1+cos(pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters))) * (max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    
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
                lr_t = lr * sqrt(1-beta2**t)
                # 1-beta1**t 直接加到denom上
                denom = v.sqrt().mul_(1-beta1**t).add_(eps)
                # ltheta -= lr_t * m / denom
                p.data.addcdiv_(m, denom, value=-lr_t)
                # theta -= lr * wd * theta
                p.data.mul_(1 - lr * wd)

                state["t"] = t + 1
        
        return loss
