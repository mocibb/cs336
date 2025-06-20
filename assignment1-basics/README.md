# CS336 Spring 2025 Assignment 1: Basics

作业一的详细说明手册参见
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)


## 跑通测试

```sh
uv run pytest
```

## 训练TinyStories数据集
下载TinyStories数据集

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

```

训练TinyStories的数据集

第一次train时需要bpe，大概1分钟左右。

同样的第一次train时也需要tokenization，大概6-7分钟。

4060ti上训练4万次迭代需要1.3个小时，最后eval数据的loss大概1.5左右。

``` sh
cd assignment1-basics

./build.sh

uv run python ./train_tinystories.py

```
 


