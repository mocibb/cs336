
import os
import time
import numpy as np
import numpy.typing as npt
import torch
from datetime import datetime
import wandb 
from tqdm import tqdm
from training import PretrainedConfig
from training import get_batch
from training import save_checkpoint
from model import TransformerLM
from optimizer import cross_entropy
from optimizer import gradient_clipping
from optimizer import AdamW, get_lr_cosine_schedule

def train(step, dataset: npt.NDArray, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config):
    # sample数据
    inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)

    # 切换到训练模式
    model.train()

    # 计算loss
    logits = model(inputs)
    loss = cross_entropy(logits, targets)

    # 梯度清零
    optimizer.zero_grad()
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
    run = wandb.init(
        project=config.project_name,
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        config=config.__dict__
    )
    print("wandb.init OK")
    
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
                          device=config.device) 

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
    print("train data size: ", train_data.shape[0], "valid data size: ", valid_data.shape[0])
    total_tokens_processed = config.batch_size*config.context_length*config.total_steps
    print("total tokens processed: ", total_tokens_processed)
    if total_tokens_processed < 327680000:
        print("warning: total_tokens_processed < 327680000, may underfit.")
    print("total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 训练循环
    start_time = time.time()
    for step in tqdm(range(1, config.total_steps+1)):
        # 更新學習率
        lr = get_lr_cosine_schedule(
            step,
            config.learning_rate,
            config.learning_rate*0.01,
            config.warmup_steps,
            config.total_steps
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # train
        loss = train(step, train_data, model, optimizer, config)

        if step % config.log_freq == 0:
            grad_norm = torch.sqrt(sum(x* x for x in [p.grad.data.norm() for p in model.parameters() if p.requires_grad]))
            wandb.log({
                'train/loss': loss, 
                'train/grad_norm': grad_norm, 
                'train/lr': lr, 
                'train/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, loss = {loss}, lr = {lr}, grad_norm = {grad_norm}")

        # 验证
        if step % config.eval_freq == 0:
            eval_loss = evaluate(valid_data, model, config)
            wandb.log({
                'val/loss': eval_loss,
                'val/wallclock_time': time.time() - start_time
            }, step=step)
            print(f"step = {step}, eval_loss = {eval_loss}")
        
        # 保存checkpoint
        if step % config.checkpoint_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt"),
            )
            print(f"Checkpoint saved to {config.checkpoint_dir}/checkpoint_{step}.pt")

    eval_loss = evaluate(valid_data, model, config)
    wandb.log({
        'val/loss': eval_loss,
        'val/wallclock_time': time.time() - start_time
    }, step=step)
    print(f"final evaluation loss: {eval_loss}")
    
    save_checkpoint(
        model,
        optimizer,
        step,
        os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
    )
    
    wandb.finish()

