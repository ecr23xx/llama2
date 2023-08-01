# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import torch
from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
):
    torch.manual_seed(1997)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )

    dialogs = []
    while True:
        print("user: ", end='', flush=True)
        msg = input()
        dialogs.append({"role": "user", "content": msg})
        print("")
        print("assistant: " , end='', flush=True)
        results = generator.chat_completion(
            [dialogs],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            stream=True
        )
        print("\n")
        dialogs.append(results[0]["generation"])

if __name__ == "__main__":
    fire.Fire(main)
