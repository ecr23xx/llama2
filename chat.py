# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import torch
import argparse
from pathlib import Path

from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max-new-tokens", default=16, type=int)
    return parser.parse_args()


def main():
    # seed must be the same in all processes
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parse_args()
    print(args)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    ckpt_dir = "./data/llama-2-7b/"
    ckpt_path = ckpt_dir + "consolidated.00.pth"
    tokenizer_path = "./data/tokenizer.model"

    tokenizer = Tokenizer(model_path=tokenizer_path)
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_args = ModelArgs(**params)
    model_args.vocab_size = tokenizer.n_words
    print(model_args)

    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    # model.to(device)

    prompt = "Hello there"
    start_ids = tokenizer.encode(prompt, bos=True, eos=False)
    x = (torch.tensor(start_ids, dtype=torch.long, device='cpu')[None, ...])
    idxs = model.generate(x, args.max_new_tokens, temperature=args.temperature)
    print(idxs)


if __name__ == "__main__":
    main()
