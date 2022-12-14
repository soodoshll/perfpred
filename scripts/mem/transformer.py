import argparse

import torch
import os
from perfpred.utils import measure_gpu_mem
from torch.cuda.amp import GradScaler

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--nitr",
        type=int,
        default=5
    )
    parser.add_argument('--amp', action="store_true")
    args = parser.parse_args()

    return args


def main():
    if 'LD_PRELOAD' in os.environ:
        use_fake_alloc = True
        import fake_alloc
        TRANSFORMER_COMPENSATE = 1024 * 1024 * 1024
        fake_alloc.set_target_mem_limit(24 * 1024 * 1024 * 1024 - TRANSFORMER_COMPENSATE)
    else:
        use_fake_alloc = False
    device = 'cuda'
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model)
    # return
    model = AutoModelForSequenceClassification.from_config(
        # args.model,
        config=config,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    inputs = torch.ones((args.batch_size, args.seq_len), dtype=torch.int64, device=device)
    labels = torch.zeros((args.batch_size, ), dtype=torch.int64, device=device)
    scaler = GradScaler(enabled=args.amp)
    def train(nitr):
        for _ in range(nitr):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        torch.cuda.synchronize()
    if use_fake_alloc:
        train(args.nitr)
        print((fake_alloc.max_mem_allocated() + TRANSFORMER_COMPENSATE) / (1024)**2)
    else:
        max_mem = measure_gpu_mem(lambda: train(args.nitr))
        print(max_mem)

if __name__ == "__main__":
    main()