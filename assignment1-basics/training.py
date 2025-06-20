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

import numpy.typing as npt
import numpy as np
import torch
import os
from typing import IO, BinaryIO
from dataclasses import dataclass

@dataclass
class PretrainedConfig():
    # project
    project_name: str
    # data parameter
    vocab_path: str
    merges_path: str
    special_tokens: list[str]
    train_path: str
    valid_path: str

    # model parameter (7.2 TinyStories)
    batch_size: int = 32 # 
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int =  1344
    rope_theta: float = 10000
    num_layers: int = 4  
    num_heads: int = 16
    use_compile: bool = True

    # training parameter (LLaMA: Open and Efficient Foundation Language Model)
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01 # 
    gradient_clipping: float = 1.0
    warmup_steps: int = 3000   # 10% of total_steps
    total_steps: int = 30000

    # logging and checkpoint
    log_freq: int = 100
    eval_freq: int = 1000
    eval_batch: int = 10
    checkpoint_freq: int = 5000
    checkpoint_dir: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # batch_size组，随机起点
    start_indices = torch.randint(0, len(dataset) - context_length, (batch_size, ))   
    # 实现扩展维度 
    total_indices = start_indices[:, None] + torch.arange(context_length + 1)

    inputs = dataset[total_indices][:, :-1]
    targets = dataset[total_indices][:, 1:]
    
    # from_numpy 本身并不会拷贝
    return tuple([torch.from_numpy(x.astype(np.int64)).to(device) for x in [inputs, targets]])

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    saved = {}
    saved['model'] = model.state_dict()
    saved['optimizer'] = optimizer.state_dict()
    saved['iteration'] = iteration 
    torch.save(saved, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer|None = None) -> int:
    saved = torch.load(src)
    try:
        model.load_state_dict(saved['model'])
    except:
        model = torch.compile(model)
        model.load_state_dict(saved['model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    return saved['iteration']
    
