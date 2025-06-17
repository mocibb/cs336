# 作业一参考资料（整理中）
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

### BPE算法
基本操作
1. 选择最高的pair
2. 合并pair，并更新pair出现次数
因为每次只合并一个pair，其影响仅仅包含当前pair的单词，更新pair出现的次数不用每次都全部更新。

#### 懒惰更新的堆结构
带标记的堆结构由headq和dict组成，可以实现O(1)的top1操作和O(1)的更新。

#### 作业（BPE）
作业中实现了一个c++的高效BPE算法，tinystories数据集上train时间不超过2秒钟，在owt数据集上小于8分钟，远远低于作业要求的12小时。

第一次运行测试前需要编译，
```bash
./build.sh
uv run pytest -k test_train_bpe
```

#### 作业（tokenizer）
Tokenizer中encode可以参考代码 [hf gpt2](https://github.com/huggingface/transformers/blob/9300728665aaeb0ebf4db99f9d9fbce916b4a183/src/transformers/models/gpt2/tokenization_gpt2.py#L187)

编写encode代码需要考虑下面情况。

考虑 Hello, 其中 el, ll都是分词得到的pair，ll比el优先级高。这时要先合并ll，因为ll已经合并，所以el不存在了。

所以encode也是每次合并一个优先级最高的pair，合并后重新计算pair和优先级。

c++的版本代码实现了单个词的encode，速度是python版的3倍左右。4个进程跑tinystories数据集大概要1分半。

如果整个tokenizer完全用c++实现应该可以快10倍。

#### 开源参考
[hg](https://github.com/huggingface/tokenizers)

[sentencepiece](https://github.com/google/sentencepiece)

## 模型

主要参考论文《LLaMA: Open and Efficient Foundation Language Models》

### Pre-vs-post Norm
post-norm要比pre-norm更难训练，post-norm往往需要加入预热处理。

论文1中定理一利用平均场理论证明了post-norm的梯度( $\mathcal{O}(d \sqrt{\ln d})$ )过大是造成难训练的原因。

论文2给出了一个新的deepnorm使得transformer结构可以达到1000层的深度。

论文3以及[苏神](https://spaces.ac.cn/archives/9009)都指出pre-norm的效果其实不如post-norm。

课程视频也介绍了新的double-norm可能平衡训练和性能的最佳方案。

参考

1. 《On Layer Normalization in the Transformer Architecture》
2. 《DeepNet: Scaling Transformers to 1,000 Layers》
3. 《Transformers without Tears: Improving the Normalization of Self-Attention》


### LayerNorm vs RMSNorm

论文1提出训练时数据移动是关键瓶颈。

这个也解释了课程视频提到的虽然从性能上RMSNorm和LayerNorm接近，但是RMSNorm减少了数据移动。

- 《Data Movement Is All You Need》
- 《Root Mean Square Layer Normalization》

### Gating, activations

### RoPE

### Hyperparameters

### Softmax稳定性问题
#### 输出Softmax稳定性z-loss
#### QK-Norm

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

### 优化器选择的经验法则 (From cs231n)
> 在许多情况下，即便不调学习率，Adam也是理想的默认选择；
> 
> 使用SGD+Momentum可能优于Adam，但通常需更精细地调整学习率及调度策略；
> 
> 如果数据集较小可以使用full batch更新的话建议尝试L-BFGS算法。


### AdamW

论文 《Decoupled Weight Decay Regularization》

AdamW是目前大语言模型主流的优化器，

**背景:**

Adam在泛化性方面被认为弱于SGD+Momentum的组合。作者认为在Adam中的L2正则化和权重衰减正则化并不等价（参见论文性质2）。

而在SGD+Momentum中L2正则化恰好等于权重衰减正则化。

权重衰减正则化被认为可以带来更好的泛化性能。 所以通过把权重衰减显式的加入到更新中提升了在CIFAR-10分类任务测试中的性能。

**L2正则化 vs 权重衰减正则化**

从这个图可以看出，L2正则化是把上一次参数经过衰减加入梯度；

而权重衰减正则化是上一次参数不经过梯度的缩放和变换直接衰减后叠加到参数。


<img src="https://github.com/user-attachments/assets/0046dd77-c890-472f-aaa1-8d57b3677362" alt="sdg vs adamw" width="400"/>


问题：

为什么权重衰减正则化的泛化性更优？

AdamW 引用了论文 《Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods》

其他关于权重衰减的论文还可以参考 《Why Do We Need Weight Decay in Modern Deep Learning?》

**训练资源计算**


### Muon
- kellerjordan的版本 https://kellerjordan.github.io/posts/muon/
- 月之暗面的开源版本 https://github.com/MoonshotAI/Moonlight

算法
<img src="https://github.com/user-attachments/assets/a89fa6fa-556c-4e02-97da-ec9b1b721316" alt="muon" width="400"/>


怎么观察梯度方向

1. 观察条件数
2. 投影到低维空间



### 学习率调度(learning_rate_tuning)

论文
- 《Llama: Open and eﬀicient foundation language models》
- 《Why Do We Need Weight Decay in Modern Deep Learning?》

## 实验

### Trick
https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135
1. 使用 numpy.memmap(), array = numpy.array(memmap_file)


https://huggingface.co/blog/train_memory
