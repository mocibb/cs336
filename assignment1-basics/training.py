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
    # 把数据转成torch
    data = torch.from_numpy(dataset).to(device)

    return data[dataset[total_indices][:, :-1]], data[dataset[total_indices][:, 1:]]

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
    