import pickle
import requests
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path

def main():
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    shakespeare_text = requests.get(data_url).text

    # Add in some Python code training data so the model learns both Shakespare and Python
    df = pd.read_parquet(
        'hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet'
    )
    python_code = '\n###\n'.join(df['output'].dropna().astype(str))
    python_code = python_code.encode('ascii', 'ignore').decode() # there's a few non-ascii characters but I don't want to deal with them

    shakespeare_text = shakespeare_text[:1_000_000]
    python_code = python_code[:1_000_000]

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    new_tokenizer = tokenizer.train_new_from_iterator(
        [shakespeare_text, python_code],
        vocab_size=1024,
    )

    val_data = python_code[int(len(python_code) * 0.9):] + shakespeare_text[int(len(shakespeare_text) * 0.9):]
    train_data = python_code[:int(len(python_code) * 0.9)] + shakespeare_text[:int(len(shakespeare_text) * 0.9)]

    # encode both to integers
    train_ids = new_tokenizer.encode(train_data)
    val_ids = new_tokenizer.encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(Path(__file__).parent / 'train.bin')
    val_ids.tofile(Path(__file__).parent / 'val.bin')

    # save the meta information as well, to help us encode/decode later
    meta = {
        # TODO: Just save the tokenizer object properly, this is jank
        'vocab_size': len(new_tokenizer.vocab),
        'encode': new_tokenizer.encode,
        'decode': new_tokenizer.decode,
    }
    with open(Path(__file__).parent / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    main()