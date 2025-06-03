# Stanford CS336 作业项目

## 介绍 
本Repo是用来提供CS336作业学习讨论的地方。

CS336是斯坦福大学在2025年开设的从零到一实现大语言模型的课程。

整个课程包括五次作业，通过五次作业可以了解到现代大语言模型的各个方面：
![image](https://github.com/user-attachments/assets/ac896cc4-3a4f-4e61-8824-8fa906b50fce)

## 作业介绍
### 作业一
* 实现BPE分词器‌
* 实现Transformer模型、交叉熵损失函数、AdamW优化器及训练循环‌
* TinyStories和OpenWebText数据集上进行训练‌
* 打榜：在H100上给定90分钟内最小化OpenWebText的困惑度‌

### 作业二
* 在Triton中实现一个融合的RMSNorm内核‌
* 实现分布式数据并行训练‌
* 实现优化器状态分片‌
* 对实现进行基准测试和性能分析‌

### 作业三
* 定义训练API标准化接口
* 提交训练任务（在FLOPs预算内）并收集训练数据
* 对训练数据拟合scaling law
* 提交对scaled up后超参数的预测结果

### 作业四
* Common Crawl HTML转文本‌
* 训练质量与安全内容分类器‌
* 基于MinHash的去重处理‌
* 打榜：在给定token预算下最小化perplexity

### 作业五
* 监督微调实现
* 直接偏好优化实现
* 群体相对偏好优化实现

## 课程资源
[课程主页](https://stanford-cs336.github.io/spring2025/)
[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
[kkaitlyn的作业](https://github.com/kkaitlyn111/cs336-assignment1)

## 如何跑起第一次作业
1. 安装uv
   参考 https://docs.astral.sh/uv/getting-started/installation/
2. 运行作业一的测试程序
```sh
uv run pytest tests/test_train_bpe.py
```
