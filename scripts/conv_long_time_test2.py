import torch
import torchvision
import subprocess
import time
from perfpred.vgg import build_vgg_model
import argparse
import os
from tqdm import trange

device = 'cuda'
batch_size = 32
# x = torch.rand(batch_size, 3, 224, 224, device=device)
x = torch.rand(batch_size, 64, 224, 224, device=device)
# model = torchvision.models.vgg13()
model = torch.nn.Conv2d(64, 64, 3)
model.to(device)
# optim = torch.optim.SGD(model.parameters(), lr=1e-3)
o = model(x)
# o = o.sum()
# o.backward()
# optim.zero_grad()
torch.cuda.synchronize()
ret = subprocess.Popen(['nvidia-smi','-lms', '50', '--query-gpu=clocks.sm', '-i','0','--format=csv,noheader,nounits', '-f', 'conv_long_clock.txt'])

nitr = 80000 
t0 = time.time()
for i in trange(nitr):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # optim.zero_grad()
    o = model(x)
    # o = o.sum()
    # o.backward()
    # optim.step()
    end.record()
    torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    print(time.time() - t0, dur)
