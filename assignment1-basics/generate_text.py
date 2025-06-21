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
import os
import torch
from tokenizer import Tokenizer
from training import load_checkpoint
from model import softmax
from model import TransformerLM
from optimizer import AdamW

def generate_text(model: TransformerLM, 
                  tokenizer: Tokenizer, 
                  prompt: str, 
                  new_tokens_limit: int = 32, temperature: float = 0.7, top_p: float = 0.9):
    
    x = torch.tensor(tokenizer.encode(prompt), device=model.device).unsqueeze(0)
    context_length = model.context_length
    for _ in range(new_tokens_limit):
        x = x[:, -context_length :] if x.size(1) > context_length else x
        # add batch
        logits = model(x)
        next_token_logits = logits[:, -1, :]
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # top-p
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        # 
        next_token_prob = softmax(sorted_logits, dim=-1)
        
        cum_probs = torch.cumsum(next_token_prob, dim=-1)
        
        mask = cum_probs <= top_p
        mask[..., 0] = True
        
        filtered_probs = torch.where(mask, cum_probs, torch.zeros_like(cum_probs))
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

        sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
        next_token_id = sorted_indices.gather(-1, sampled_indices)

        if (next_token_id[0, 0] == tokenizer.eos_id):
            break

        x = torch.cat((x, next_token_id), dim=-1)

    return tokenizer.decode(x.squeeze(0).tolist())


if __name__ == '__main__':
    root_folder = os.path.dirname(os.path.abspath(__file__))
    checkpoint_folder = "{}/saved/".format( root_folder )

    model = TransformerLM(vocab_size=10000, 
                          context_length=128,
                          d_model=512,
                          num_layers=4,
                          num_heads=16,
                          d_ff=1344,
                          rope_theta=10000,
                          device="cuda") 
    model.to("cuda")
    load_checkpoint(f"{checkpoint_folder}/checkpoint_512.pt", model)
    model.eval()

    tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                     f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                     special_tokens=['<|endoftext|>'])
    prompt = "Once upon a time, there was a girl named Sue."
    text = generate_text(model, tokenizer, prompt, 96)
    print(text)
