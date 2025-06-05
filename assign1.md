# 作业一参考资料
## 分词(Tokenizer)
分词指标包括，词汇数目(vocabulary size)和压缩率(compression ratio)。

词汇数目影响模型的大小，常见的大模型单语言一般规模在5万以内，多语言一般大于10万。作业一的词汇数目是1万。

压缩率 = 字节数/token数。压缩率通常是在指定语料上统计，压缩率越高越好。同样一句话"跟我学习语文"会被ds分词成"跟我", "学习"和"语文"，而被gpt-4o分成"跟","我","学习","语"和"文"，所以这个例子上ds的压缩率更高。大模型一般通过计算token数目收费，所以分词的api和分词的json文件都会公开，但是算法一般不会公开。ds的分词json文件可以在[这里](https://api-docs.deepseek.com/quick_start/token_usage)查看。

一个简单的想法就是用自然单词作为分词，但是缺点是自然单词出现的频率不均匀，有的词例如"觥筹"过于稀少，而相反有些不是自然词却更多，比如"跟我"。ds就会把"跟我来"，分成"跟我"和"来"。

常见算法包括: BPE，Unigram和WordPiece。

- [openai](https://platform.openai.com/tokenizer)
- [tiktokenizer比较](https://tiktokenizer.vercel.app/)
- [deepseek在线](https://lunary.ai/deepseek-tokenizer) 

中文分词算法


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
