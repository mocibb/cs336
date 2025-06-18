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
import torch
import os
from typing import IO, BinaryIO

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # batch_size组，随机起点
    start_indices = torch.randint(0, len(dataset) - context_length, (batch_size, ))   
    # 实现扩展维度 
    total_indices = start_indices[:, None] + torch.arange(context_length + 1)

    inputs = dataset[total_indices][:, :-1]
    targets = dataset[total_indices][:, 1:]

    return tuple([torch.from_numpy(x).type(torch.int64).to(device) for x in [inputs, targets]])

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes], run_info: dict = None):
    saved = {}
    saved['model'] = model.state_dict()
    saved['optimizer'] = optimizer.state_dict()
    saved['iteration'] = iteration 
    if run_info is not None:
        saved['run_info'] = run_info
    torch.save(saved, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    saved = torch.load(src)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    return saved['iteration']
    
