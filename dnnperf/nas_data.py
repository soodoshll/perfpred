import torch
import argparse
import xautodl
from xautodl.models import get_cell_based_tiny_net
from nats_bench import create

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
config = api.get_net_config(args.model_id, 'cifar10')
model = get_cell_based_tiny_net(config)
model.to(device)

model.train()

inputs = torch.rand((args.batch_size, 3, args.image_size, args.image_size), device=device)
labels = torch.zeros((args.batch_size,), dtype=torch.int64, device=device)

loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_loop():
    optim.zero_grad()
    out = model(inputs)[0]
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()

train_loop() # benchmark

torch.cuda.synchronize()

# measure iteration time
t0 = time.time()
train_loop()
iteration_time = time.time() - t0

def measure_gpu_mem(max_mem, tot_time=10, interval=0.1):
    n = max(10, int(tot_time / interval))
    max_mem.value = 0
    for i in range(n):
        ret = subprocess.run(['nvidia-smi','--query-gpu=memory.used', '-i','0','--format=csv,noheader,nounits'], capture_output=True)
        mem = float(ret.stdout)
        max_mem.value = max(max_mem.value, mem)
        time.sleep(interval)

max_mem = multiprocessing.Value('f', 0.0)
measure_proc = multiprocessing.Process(target=measure_gpu_mem, args=(max_mem, 3 * iteration_time))
measure_proc.start()

for i in range(3):
    train_loop()

measure_proc.join()
print(max_mem.value)
exit(0)