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

### BPE算法
基本操作
1. 选择最高的pair
2. 合并pair，并更新pair出现次数

#### 带标记的堆结构
带标记的堆结构由headq和dict组成，可以实现O(1)的top1操作和O(1)的更新。

## 模型
### Pre-vs-post Norm
减小梯度Norm的跳动
### LayerNorm vs RMSNorm
1. Data Movement Is All You Need
2. Root Mean Square Layer Normalization
### Gating, activations

### RoPE

### Hyperparameters

### Softmax稳定性问题
#### 输出Softmax稳定性z-loss
#### QK Norm

### Attention heads

#### GQA / MQA (KV Cache)

#### 

### The Full Transformer LM资源计算表
[计算器](https://docs.google.com/spreadsheets/d/1LebxBI5lkoNdMFEBIOIEnHylSvvzoC8xvWMBcXwjy7U/edit?usp=sharing)

## 训练

### Cross-entropy loss
交叉熵的输入inputs包含下一次要接龙的单词的概率，它是由我们模型得到的预测值，注意是没有过Softmax。 

inputs的大小是 $D\times V$ 张量，其中 $D$ 是batch数目， $V$ 为词汇数目。通过softmax给出由inputs预测下一个单词的概率为，

$p_\theta\left(x_{i+1} \mid x_{1: i} \right)=\frac{\exp\left( \text{input}[x_i] \right)}{\sum_a \exp\left(\text{input}[a]\right) }$

交叉熵另一个输入项targets是下一次要接龙的目标单词，记为 $\hat{x}\_{i+1}$

计算数据集的总交叉熵为，

$\ell(\theta ; D)=-\frac{1}{|D|} \sum_{x \in D} \log p_\theta\left(x_{i+1} = \hat{x}\_{i+1} \mid x_{1: i}\right) $

### AdamW
论文 《Decoupled Weight Decay Regularization》

优点：

原理：


### Muon
https://kellerjordan.github.io/posts/muon/

### 学习率调度(learning_rate_tuning)

## 实验

### Trick
https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135
1. 使用 numpy.memmap(), array = numpy.array(memmap_file)
2. 
