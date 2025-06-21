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

4060ti上训练4万次迭代需要1.5个小时，最后eval数据的loss大概1.45左右。

``` sh
cd assignment1-basics

./build.sh

uv run python ./train_tinystories.py

```
最后附上一段训练后写的作文，生成作文的代码参考generate_text.py

>Once upon a time, there was a girl named Sue. Sue had a pretty doll that she loved very much. She took her doll everywhere she went. One day, Sue's doll broke. Sue was very sad.
>Sue's mom saw her sad face and asked, "What's wrong, Sue?" Sue showed her mom the broken doll. Her mom said, "Don't worry, we can replace your doll."
>Sue and her mom went to the store. They bought some ice cream. Sue was happy.


