import argparse

import torch
import os
from perfpred.utils import measure_gpu_mem, remove_initialization
from torch.cuda.amp import GradScaler

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset

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
    parser.add_argument(
        '--checkpoint',
        action='store_true'
    )
    parser.add_argument('--amp', action="store_true")
    args = parser.parse_args()
    return args


def main():
    if 'LD_PRELOAD' in os.environ:
        use_fake_alloc = True
        import fake_alloc
        TRANSFORMER_COMPENSATE = 1024 * 1024 * 1024
        # fake_alloc.set_target_mem_limit(80 * 1024 * 1024 * 1024)
        remove_initialization()
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
    print("model created")
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    inputs = torch.ones((args.batch_size, args.seq_len), dtype=torch.int64, device=device)
    labels = torch.zeros((args.batch_size, ), dtype=torch.int64, device=device)
    dataset = Dataset.from_dict({'input_ids':inputs, 'labels':labels})
    # scaler = GradScaler(enabled=args.amp)
    scaler = GradScaler(enabled=False)
    # DO NOT USE GRAD SCALER!!!!

    def train(args):
        training_args = TrainingArguments(
            "./tmp/", 
            per_device_train_batch_size=args.batch_size, 
            gradient_checkpointing=args.checkpoint,
            num_train_epochs=args.nitr,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        # for _ in range(args.nitr):
            # trainer.training_step(model, {'input_ids':inputs, 'labels':labels}) 
        trainer.train()
        torch.cuda.synchronize()
    if use_fake_alloc:
        train(args)
        print((torch.cuda.max_memory_reserved() + TRANSFORMER_COMPENSATE) / (1024)**2)
        # print(fake_alloc.max_mem_allocated(), torch.cuda.max_memory_reserved(), torch.cuda.max_memory_allocated())
    else:
        max_mem = measure_gpu_mem(lambda: train(args))
        print(max_mem, torch.cuda.max_memory_reserved(), torch.cuda.max_memory_allocated())

if __name__ == "__main__":
    main()