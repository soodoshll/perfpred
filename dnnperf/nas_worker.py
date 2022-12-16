import torch
import argparse
import xautodl
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create

from perfpred.utils import measure_gpu_mem

import time
import multiprocessing, subprocess


from trace import Graph

parser = argparse.ArgumentParser()

parser.add_argument("model_id", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("image_size", type=int)

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = 'cuda'

api = create(None, 'tss', fast_mode=True, verbose=False)
config = api.get_net_config(args.model_id, 'ImageNet16-120')
model = get_cell_based_tiny_net(config)
model.to(device)

model.train()

inputs = torch.rand((args.batch_size, 3, args.image_size, args.image_size), device=device)
labels = torch.zeros((args.batch_size,), dtype=torch.int64, device=device)

loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_loop(nitr=3):
    for _ in range(nitr):
        optim.zero_grad()
        out = model(inputs)[0]
        loss = loss_fn(out, labels)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

train_loop() # benchmark

max_mem = measure_gpu_mem(train_loop)

print(max_mem)
exit(0)