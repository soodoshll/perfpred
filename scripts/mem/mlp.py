import torch
from torch import nn
import argparse
from pyinstrument import Profiler
from torch.nn.utils import skip_init
from perfpred.utils import remove_initialization, measure_gpu_mem
import os
import subprocess
from subprocess import PIPE, STDOUT

parser = argparse.ArgumentParser()
parser.add_argument('--ndim', type=int, default=1024)
parser.add_argument('--nlayers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--nitr', type=int, default=2)
args = parser.parse_args()

if 'LD_PRELOAD' in os.environ:
    use_fake_alloc = True
    import fake_alloc
else:
    use_fake_alloc = False

remove_initialization()

def train(nitr=2):
    def make_mlp(input_dim, hidden_layers=[1024] * 8  , activation=nn.ReLU):
        layers = []
        last = input_dim
        for _, h in enumerate(hidden_layers):
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        model = nn.Sequential(*layers)
        return model
    fake_alloc.set_target_mem_limit(24 * 1024 * 1024 * 1024)
    fake_alloc.reset_max_mem()

    device = 'cuda:0'
    model = make_mlp(args.ndim, [args.ndim]*args.nlayers)
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.empty((args.batch_size, args.ndim), device=device)

    for _ in range(nitr):
        optim.zero_grad()
        out = model(x)
        out = out.sum()
        out.backward()
        optim.step()
    
    torch.cuda.synchronize()

if os.path.isfile("log_mem.tmp"):
    os.remove("log_mem.tmp")

pid = os.getpid()

if use_fake_alloc:
    train(args.nitr)
    print(fake_alloc.max_mem_allocated() / (1024)**2)
else:
    monitor_proc = subprocess.Popen(['nvidia-smi', '-lms', '10', '--query-compute-apps=pid,used_gpu_memory', '--format=csv', '-f', 'log_mem.tmp'])
    train(args.nitr)
    monitor_proc.terminate()
    max_mem = 0
    with open("log_mem.tmp", "r") as f:
        for line in f:
            tokens = line.split(',')
            if int(tokens[0]) == pid:
                mem = int(tokens[1].split()[0])
                max_mem = max(max_mem, mem)
    os.remove("log_mem.tmp")
    print(max_mem)
# ret = monitor_proc.communicate()[0]
# print(len(ret))
