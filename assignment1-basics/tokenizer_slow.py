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
import pickle
import os
import functools
import numpy as np
from abc import ABC
import multiprocessing
import heapq
from collections.abc import Iterable, Iterator
from cs336_basics.pretokenization import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    vocab: dict[int, bytes]
    inv_vocab: dict[bytes, int]
    special_tokens_pattern: str | None

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inv_vocab = {}
        for k, v in vocab.items():
            self.inv_vocab[v] = k

        self.special_tokens_pattern = None
        if special_tokens:
            self.special_tokens_pattern = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = []
        start = 0
        if self.special_tokens_pattern:
            for m0 in re.finditer(self.special_tokens_pattern, text):
                
                for match in re.finditer(PAT, text[start:m0.start()]):
                    tokens.extend(self._tokenize(match.group()))
                tokens.append(self.inv_vocab[m0.group().encode("utf-8")])
                start = m0.end()

        if start < len(text):
            for match in re.finditer(PAT, text[start:]):
                tokens.extend(self._tokenize(match.group()))
        return tokens
    
    def encode_chunk(self, chuck: tuple[int], input_path: str) -> list[int]:
        start, end = chuck
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return self.encode(chunk)
        return None
    
    def encode_file(self, input_path: str, output_path: str, num_split: int = 4, num_processes: int=4) -> None:
        boundaries = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_split, "<|endoftext|>".encode("utf-8"))
        
        all_token_ids = []
        t0 = time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(
                    functools.partial(self.encode_chunk, input_path=input_path),
                    zip(boundaries[:-1], boundaries[1:]),
                )
            
            for res in results:
                all_token_ids.extend(res)
            
        t1 = time.time()

        print("encode_file: ", t1-t0)

        np.save(output_path, np.array(all_token_ids, dtype=np.uint16))


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            # Encode each string and yield all its token IDs
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        text_bytes = []
        for id in ids:
            text_bytes.append(self.vocab[id])
        return b''.join(text_bytes).decode('utf-8', errors='replace')

    # 加速
    def _tokenize(self, text: str) -> list[int]:
        word_list = [bytes([b]) for b in text.encode('utf-8')]
        while True:
            merge_candidates = []
            for i in range(len(word_list)-1):
                try:
                    pair_bytes = b''.join(word_list[i:i+2])
                    merge_candidates.append( (self.inv_vocab[pair_bytes], i, pair_bytes) )
                except KeyError:
                    pass
        
            if len(merge_candidates) == 0:
                break
            best_merge = min( merge_candidates )
            idx = best_merge[1]
            word_list[idx:idx+2] = [best_merge[2]]
    
        return [self.inv_vocab[b] for b in word_list]


if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.abspath(__file__))

    tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                     f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                     special_tokens=['<|endoftext|>'])


    tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy')
    # tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy')
    # tokenizer.encode_file(f'{root_folder}/data/corpus.en', f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy')
    
