
import os
import time
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import wandb 

from dataclasses import dataclass
from tqdm import tqdm
from training import get_batch
from training import save_checkpoint
from model import TransformerLM
from optimizer import cross_entropy
from optimizer import gradient_clipping
from optimizer import AdamW, get_lr_cosine_schedule

from tokenizer import Tokenizer
 
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
    d_ff: int = 1344
    rope_theta: float = 10000
    num_layers: int = 4
    num_heads: int = 16
    use_compile: bool = True

    # training parameter (LLaMA: Open and Efficient Foundation Language Model)
    learning_rate: float = 3e-4*10
    beta1: float = 0.90
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clipping: float = 5.0
    warmup_steps: int = 270
    max_steps: int = 10000

    # logging and checkpoint
    log_freq: int = 100
    eval_freq: int = 1000
    eval_batch: int = 10
    checkpoint_freq: int = 5000
    checkpoint_dir: str | None = None

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataset: npt.NDArray, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config):
    # sample数据
    inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
    
    # 切换到训练模式
    model.train()

    # 梯度清零
    optimizer.zero_grad()

    # 计算loss
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    # 梯度下降，自动求导
    loss.backward()
    gradient_clipping(model.parameters(), config.gradient_clipping)

    # 更新模型
    optimizer.step()

    return loss.item()

def evaluate(dataset: npt.NDArray, model: torch.nn.Module, config):
    # 切换到eval模式
    model.eval()

    losses = []
    with torch.no_grad():
        for n in range(config.eval_batch):
            inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_model(config : PretrainedConfig):
    # setup logger
    # run = wandb.init(
    #     project=config.project_name,
    #     name=datetime.now().strftime("%Y%m%d_%H%M%S")
    # )

    device = torch.device(config.device)
    
    # 创建checkpoint文件夹
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = torch.device(config.device)

    #设置PyTorch中乘法精度
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    # 加载数据
    train_data = np.memmap(config.train_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(config.valid_path, dtype=np.uint16, mode='r')
    # 
    # 模型
    model = TransformerLM(vocab_size=config.vocab_size, 
                          context_length=config.context_length,
                          d_model=config.d_model,
                          num_layers=config.num_layers,
                          num_heads=config.num_heads,
                          d_ff=config.d_ff,
                          rope_theta=config.rope_theta,
                          device=device)
    model = model.to(device) 

    if config.use_compile:
        print("Compiling model for better performance...")
        model = torch.compile(model)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2)
    )
    
    print("train device: ", config.device)
    print("train data size: ", train_data.shape, "valid data size: ", valid_data.shape)
    # Downscaling tip: 40, 000, 000
    print("total tokens processed: ", config.batch_size*config.context_length*config.max_steps)
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    loss_history = []  
    # 训练循环
    for epoch in tqdm(range(config.max_steps)):
        # train
        loss = train(train_data, model, optimizer, config)
        print(f"train loss: {loss}")

        lr = get_lr_cosine_schedule(
            epoch,
            config.learning_rate*0.01,
            config.learning_rate,
            config.warmup_steps,
            config.max_steps
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 验证
        if epoch % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            print(f"evaluation loss: {eval_loss}")
        
        # 保存checkpoint
        if epoch % config.checkpoint_freq == 0:
            run_info = {
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())            
            }
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(config.checkpoint_dir, f"checkpoint_{epoch}.pt"),
                run_info=run_info
            )
    
            print(f"Checkpoint saved to {config.checkpoint_dir}/checkpoint_{epoch}.pt")

    evaluate(valid_data, model, config)
    print(f"evaluation loss: {eval_loss}")
    
    save_checkpoint(
        model,
        optimizer,
        epoch,
        os.path.join(config.checkpoint_dir, f"checkpoint_{epoch}.pt"),
        run_info=run_info
        )
if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = "{}/data/".format( root_folder )
    checkpoint_folder = "{}/checkpoints/".format( root_folder )
    
    config = PretrainedConfig(
        project_name="tinystories",
        vocab_path=f"{data_folder}/TinyStoriesV2-vocab.pkl",
        merges_path=f"{data_folder}/TinyStoriesV2-merges.pkl",
        special_tokens=["<|endoftext|>"],
        train_path=f"{data_folder}/TinyStoriesV2-GPT4-train.npy",
        valid_path=f"{data_folder}/TinyStoriesV2-GPT4-valid.npy",
        checkpoint_dir=checkpoint_folder
    )

    train_model(config)
