# 作业一参考资料
## 分词(Tokenizer)
分词指标包括，词汇数目(vocabulary size)和压缩率(compression ratio)

压缩率 = 字节数/token数 

常见算法包括: BPE，Unigram和WordPiece。

- [openai](https://platform.openai.com/tokenizer)
- [tiktokenizer比较](https://tiktokenizer.vercel.app/)
- [deepseek在线](https://lunary.ai/deepseek-tokenizer) [deepseek官方](https://api-docs.deepseek.com/quick_start/token_usage)

词汇大小
<img src="https://github.com/user-attachments/assets/4526866f-c433-4f4b-8e66-da5c7e25b8f6" alt="vocab" width="600"/>


## 模型
### RMSNorm

### SwiGLU


### Serial vs Parallel layers

### RoPE

### Hyperparameters

### Dropout and other regularization

### GQA/MQA

### The Full Transformer LM资源计算表
[计算器](https://docs.google.com/spreadsheets/d/1LebxBI5lkoNdMFEBIOIEnHylSvvzoC8xvWMBcXwjy7U/edit?usp=sharing)

## 训练

### The SGD Optimizer

### AdamW

### Muon
https://kellerjordan.github.io/posts/muon/

### 学习率调度(learning_rate_tuning)

## 实验

### Trick
https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135
1. 使用 numpy.memmap(), array = numpy.array(memmap_file)
2. 
