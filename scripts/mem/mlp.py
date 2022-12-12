import torch
from torch import nn
import argparse
from torch.nn.utils import skip_init
from perfpred.utils import remove_initialization, parse_mem_log
import os
import subprocess
from subprocess import PIPE, STDOUT
import time

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

    device = 'cuda:0'
    model = make_mlp(args.ndim, [args.ndim]*args.nlayers)
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.empty((args.batch_size, args.ndim), device=device)

    for _ in range(nitr):
        optim.zero_grad(set_to_none=False)
        out = model(x)
        out = out.sum()
        out.backward()
        optim.step()
    
    torch.cuda.synchronize()


if use_fake_alloc:
    fake_alloc.set_target_mem_limit(128 * 1024 * 1024 * 1024)
    fake_alloc.reset_max_mem()
    train(args.nitr)
    print(fake_alloc.max_mem_allocated() / (1024)**2)
else:
    if os.path.isfile("log_mem.tmp"):
        os.remove("log_mem.tmp")
    monitor_proc = subprocess.Popen(['nvidia-smi', '-lms', '10', '--query-compute-apps=pid,used_gpu_memory', '--format=csv', '-f', 'log_mem.tmp'])
    train(args.nitr)
    monitor_proc.terminate()
    time.sleep(1)
    max_mem = parse_mem_log("log_mem.tmp", os.getpid())
    os.remove("log_mem.tmp")
    print(max_mem)
# ret = monitor_proc.communicate()[0]
# print(len(ret))
