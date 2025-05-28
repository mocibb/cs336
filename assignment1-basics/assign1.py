import time
import regex as re
from collections import Counter, defaultdict
from abc import ABC
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
import multiprocessing
import functools
from typing import BinaryIO
import os
import heapq
import io
from priority_dict import PriorityDict


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

def initialize_occurrence_pair(vocab:dict[tuple[bytes], int]) -> tuple[defaultdict[tuple[bytes], int], 
                                                                       defaultdict[tuple[bytes], set[tuple[bytes]]]]:
    """
    Initialize the occurrence pair dictionary.
    """
    pairsfreq = PriorityDict()
    pair2symbols = defaultdict(set)
    for symbols, freq in vocab.items():
        for i in range(len(symbols)-1):
            pairsfreq[symbols[i], symbols[i+1]] += freq
            pair2symbols[symbols[i], symbols[i+1]].add(symbols)
    return pairsfreq, pair2symbols
    
def process_chunk(chuck: tuple[int], 
                  input_path: str, 
                  special_tokens: list[str]) -> Counter:
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
                if match.group():
                    chunk_counter.update([tuple(bytes([b]) for b in match.group().encode('utf-8'))])
    return chunk_counter

# 维护三个数据结构
# 1. vocab_counter: symbol的词频数目
# 2. occur_pair_freq: pair的词频数目 
# 3. pair2symbols: pair对应的symbol
def train_bpe(input_path: str, 
              vocab_size: int, 
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 0. 初始变量
    num_processes = 2
    # 1. 构建初始词汇表
    vocab : dict[int, bytes] = { k: bytes([k]) for k in range(256) }
    for (i, special_token) in enumerate(special_tokens):
        vocab.update( { 256+i: special_token.encode() } )
    initial_vocab_size = len(vocab)
    merges : list[tuple[bytes, bytes]] = []
    vocab_counter : dict[tuple[bytes], int] = Counter()
    
    # 2. 预分词(pre-tokenization)
    t0 = time.time()
    boundaries = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
    t1 = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(
                functools.partial(process_chunk, input_path=input_path, special_tokens=special_tokens), 
                zip(boundaries[:-1], boundaries[1:]),
            )
        
        for res in results:
            vocab_counter.update(res)
    t2 = time.time()
    # 3. 训练BPE
    def update_vocab(pair):
        new_elem = (pair[0])+(pair[1])
        vocab.update({len(vocab): new_elem})
        merges.append(pair)
    
    occur_pair_freq, pair2symbols = initialize_occurrence_pair(vocab_counter)

    def merge_pair(symbol: tuple[bytes], pair:tuple[bytes], pair_bytes: bytes, freq: int) -> tuple[bytes]:
        ret = []
        i = 0
        n = len(symbol)
        remove_pairs = []

        while i < n:
            if i < n - 1:
                remove_pairs.append((symbol[i], symbol[i+1]))
            
            if i < n - 1 and (symbol[i], symbol[i+1]) == pair:
                if i > 0:
                    occur_pair_freq[symbol[i-1], symbol[i]] -= freq
                    occur_pair_freq[symbol[i-1], pair_bytes] += freq
                if i < n - 2:
                    occur_pair_freq[symbol[i+1], symbol[i+2]] -= freq
                    occur_pair_freq[pair_bytes, symbol[i+2]] += freq
                ret.append(pair_bytes)
                i += 2 
            else:
                ret.append(symbol[i])
                i += 1
            
        new_symbol = tuple(ret)

        for i in range(len(new_symbol)-1):
            pair2symbols[new_symbol[i], new_symbol[i+1]].add(new_symbol)
        for p in remove_pairs:
            pair2symbols[p].discard(symbol)
        return new_symbol

    print("occur_pair:", len(occur_pair_freq))
    print("vocab_counter:", len(vocab_counter))
    return None
    t3 = time.time()
    best_pair, _ = occur_pair_freq.pop()
    while len(vocab) < vocab_size:
    # for i in range(6):
        # print("best_pair:", best_pair)
        # print("vocab_counter:", vocab_counter)
        new_vocab = {}
        del_keys = set()

        best_pair_bytes = best_pair[0] + best_pair[1]
        symbols_to_merge = pair2symbols.pop(best_pair)
        # print("symbols:", symbols)
        for symbol in symbols_to_merge:
            freq = vocab_counter[symbol]
            new_symbol = merge_pair(symbol, best_pair, best_pair_bytes, freq)
            new_vocab[new_symbol] = freq
            del_keys.add(symbol)
        
        vocab_counter.update(new_vocab)
        for k in del_keys:
            del vocab_counter[k]
        update_vocab(best_pair)
        best_pair, _ = occur_pair_freq.pop()

    t4 = time.time()
    print("read = ", t1-t0, ", pre-tokenization = ", t2-t1, ", pre-train = ", t3-t2, ", merge = ", t4-t3)
    return vocab, merges


if __name__ == "__main__":
    # train_bpe('/home/mocibb/cs336/assignment1-basics/bpe_text.txt', 10, ['<|endoftext|>'])
    train_bpe('/home/mocibb/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
