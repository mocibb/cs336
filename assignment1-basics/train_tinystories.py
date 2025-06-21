
import os
from train_bpe import train_bpe
from tokenizer import Tokenizer
from training import PretrainedConfig
from train_loop import train_model

if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-vocab.pkl'):
        train_bpe(f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'],
              False, f'{root_folder}/data/TinyStoriesV2-pretokens.pkl',
              [f'{root_folder}/data/TinyStoriesV2-vocab.pkl', f'{root_folder}/data/TinyStoriesV2-merges.pkl'])
        print("finish bpe for tinystories.")

    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy'):
        tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                        f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                        special_tokens=['<|endoftext|>'])
        tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-train.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-train.npy')
        print("finish tokenization for train dataset of tinystories.")

    if not os.path.exists(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy'):
        tokenizer = Tokenizer.from_files(f'{root_folder}/data/TinyStoriesV2-vocab.pkl', 
                                        f'{root_folder}/data/TinyStoriesV2-merges.pkl', 
                                        special_tokens=['<|endoftext|>'])
        tokenizer.encode_file(f'{root_folder}/data/TinyStoriesV2-GPT4-valid.txt', f'{root_folder}/data/TinyStoriesV2-GPT4-valid.npy')
        print("finish tokenization for valid dataset of tinystories.")

    checkpoint_folder = "{}/checkpoints/".format( root_folder )
    data_folder = "{}/data/".format( root_folder )
    
    config = PretrainedConfig(
        project_name="tinystories",
        vocab_path=f"{data_folder}/TinyStoriesV2-vocab.pkl",
        merges_path=f"{data_folder}/TinyStoriesV2-merges.pkl",
        special_tokens=["<|endoftext|>"],
        train_path=f"{data_folder}/TinyStoriesV2-GPT4-train.npy",
        valid_path=f"{data_folder}/TinyStoriesV2-GPT4-valid.npy",
        checkpoint_dir=checkpoint_folder
    )

    # train_model(config)
    print("finish train for tinystories.")
    
