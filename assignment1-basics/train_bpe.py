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
from collections import Counter, defaultdict
import multiprocessing
import functools
from typing import BinaryIO
import os
import pickle
from priority_dict import PriorityDict
from bpe import BytePairEncoding

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

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
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 0. 初始变量
    num_processes = 8
    num_split = 1024
    vocab_counter : dict[bytes, int] = Counter()

    # 1. 预分词(pre-tokenization)
    t0 = time.time()
    boundaries = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_split, "<|endoftext|>".encode("utf-8"))
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(
                functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens),
                zip(boundaries[:-1], boundaries[1:]),
            )

        for res in results:
            vocab_counter.update(res)
    
    t1 = time.time()
    print("pre-tokenization = ", t1-t0)
    
    bpe = BytePairEncoding(vocab_size, special_tokens)
    bpe.setVocabFreq(vocab_counter)
    result = bpe.merge()
    t2 = time.time()

    print("train = ", t2-t1)

    return result

if __name__ == "__main__":
    # train_bpe('/home/mocibb/cs336/assignment1-basics/bpe_text.txt', 10, ['<|endoftext|>'])
    train_bpe('/home/mocibb/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
    # train_bpe('/home/mocibb/cs336/assignment1-basics/data/owt_train.txt', 32000, ['<|endoftext|>'])
