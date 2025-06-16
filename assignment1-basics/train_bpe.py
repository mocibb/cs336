# coding=utf-8
# Copyright (c) 2025 mocibb (mocibb@163.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import regex as re
from collections import Counter
import multiprocessing
import functools
import os
import pickle
from bpe import BytePairEncoding
from cs336_basics.pretokenization import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(chuck: tuple[int],
                  input_path: str,
                  special_tokens: list[str],
                  strip = False) -> Counter:
    """
    Process each chunk of the file and update the vocabulary counter.
    """
    start, end = chuck
    special_tokens_pattern = '|'.join(special_tokens)
    # special_tokens_pattern = "|".join(map(re.escape, special_tokens))
    chunk_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 2.1 在预分词前移除特殊标记
        for segment in re.split(special_tokens_pattern, chunk):
            # 3. 预分词(pre-tokenization)
            for match in re.finditer(PAT, segment):
                token = match.group()
                if strip:
                    token = token.strip()
                if token:
                    chunk_counter.update([token.encode('utf-8')])
    return chunk_counter

# 维护三个数据结构
# 1. vocab_counter: word的词频数目
def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str],
              use_pretoken_file: bool = False,
              pretoken_file: str | None = None,
              bpe_results_file: list[str] | None = None
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 0. 初始变量
    num_processes = 8
    num_split = 1024
    vocab_counter : dict[bytes, int] = Counter()

    # 1. 预分词(pre-tokenization)
    t0 = time.time()

    if use_pretoken_file:
        with open(pretoken_file, "rb") as f:
            vocab_counter = pickle.load(f)
    else:
        boundaries = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_split, "<|endoftext|>".encode("utf-8"))
        print("boundaries ok")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(
                    functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
                    zip(boundaries[:-1], boundaries[1:]),
                )

            for res in results:
                vocab_counter.update(res)
        
        if pretoken_file:
            with open(pretoken_file, "wb") as f:
                pickle.dump(vocab_counter, f)

    t1 = time.time()
    print("pre-tokenization = ", t1-t0)
    
    bpe = BytePairEncoding(vocab_size, special_tokens)
    bpe.setVocabFreq(vocab_counter)
    result = bpe.merge()
    t2 = time.time()

    print("train = ", t2-t1)

    if bpe_results_file:
        with open(bpe_results_file[0], "wb") as f:
            pickle.dump(result[0], f)
        with open(bpe_results_file[1], "wb") as f:
            pickle.dump(result[1], f)

    return result

if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.abspath(__file__))

    train_bpe(f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'],
              True, f'{root_folder}/data/TinyStoriesV2-pretokens.pkl',
              [f'{root_folder}/data/TinyStoriesV2-vocab.pkl', f'{root_folder}/data/TinyStoriesV2-merges.pkl'])
