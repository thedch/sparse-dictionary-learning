import torch
import logging
import argparse
from resource_loader import ResourceLoader

def main(gpt_ckpt_dir: str, prompt: str):
    resourceloader = ResourceLoader(
        dataset='shakespeare_char',
        gpt_ckpt_dir=gpt_ckpt_dir,
        device='cpu',
        mode="prepare",
    )
    enc_fxn, dec_fxn = resourceloader.load_tokenizer()
    tokens = torch.Tensor([enc_fxn(prompt)]).long()
    logging.info(tokens)
    generated = resourceloader.transformer.generate(
        idx=tokens,
        max_new_tokens=100,
    )
    generated = dec_fxn(generated.squeeze().tolist())
    print(generated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt_ckpt_dir', type=str, default='')
    parser.add_argument('--prompt', type=str, help='Try "def run(" or "oh romeo!"')
    args = parser.parse_args()
    main(**vars(args))